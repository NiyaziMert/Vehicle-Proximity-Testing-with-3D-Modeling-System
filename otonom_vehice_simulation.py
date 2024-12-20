import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ultralytics import YOLO
import time
from google.colab import files

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# MiDaS derinlik modelini yükle
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# MiDaS için uygun giriş dönüşümü
transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize ekledik
])

# Kalibrasyon için referans mesafe
reference_distance_m = 2.0  # Referans mesafe (örneğin, 2 metre)
reference_avg_depth = None
scale_factor = 1.0  # Ön tanımlı bir değer veriyoruz

cap = cv2.VideoCapture(0)

# Hedef sınıflar
vehicle_classes = ['car', 'truck', 'bus']  # Araç sınıfları
ignored_classes = ['road']  # Yol hariç tutulacak sınıflar

# Fonksiyonlar
def create_point_cloud(points, colors):
    """ Nokta bulutu oluşturma """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def create_cube(center, size=0.5):
    """ Küp modeli oluşturma """
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.translate(center)
    mesh.paint_uniform_color([1, 0, 0])  # Kırmızı renk
    return mesh

def create_sphere(center, radius=0.5):
    """ Küre modeli oluşturma """
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center)
    mesh.paint_uniform_color([0, 0, 1])  # Mavi renk
    return mesh

# Gerçek zamanlı işlem döngüsü
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya dosya bulunamadı!")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)  # YOLOv8 ile nesne tespiti
    point_cloud_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ignored_classes:
                continue

            roi = image_rgb[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (384, 384))
            roi_rgb = torch.from_numpy(roi_resized).float().div(255).permute(2, 0, 1)  # Tensor dönüşümü ve normalize etme
            input_batch = transform(roi_rgb.unsqueeze(0))

            with torch.no_grad():
                depth = midas(input_batch)  # Modelin tahmini
                depth_matrix = depth.squeeze().cpu().numpy()

            h, w = roi.shape[:2]
            depth_matrix = cv2.resize(depth_matrix, (w, h))
            min_depth = depth_matrix.min()
            max_depth = depth_matrix.max()

            if max_depth > min_depth:
                depth_matrix = (depth_matrix - min_depth) / (max_depth - min_depth)  # 0-1 aralığına normalize et

            avg_depth = np.mean(depth_matrix)

            if label in vehicle_classes and avg_depth > 0:  # Bu kontrolle scale_factor'in sıfır bölmesini önleriz
                reference_avg_depth = avg_depth
                scale_factor = reference_distance_m / reference_avg_depth
                print(f"Kalibrasyon Güncellendi: Ölçek Faktörü = {scale_factor:.4f}")

            real_distance = (avg_depth / scale_factor) * 100
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_z = real_distance

            if real_distance < 1.0 and label in vehicle_classes:
                print(f"ALARM: {label} 1 metre altına yaklaştı! Uzaklık: {real_distance:.2f} m")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                ss_path = f"screenshot_{timestamp}.png"
                cv2.imwrite(ss_path, frame)
                print(f"Ekran görüntüsü kaydedildi: {ss_path}")

                if label in vehicle_classes:
                    sphere = create_sphere([center_x, center_y, center_z])
                    point_cloud_objects.append(sphere)
                else:
                  cube = create_cube([center_x, center_y, center_z])
                  point_cloud_objects.append(cube)
                    
                if point_cloud_objects:
                    print("3D Model Oluşturuluyor...")
                    o3d.visualization.draw_geometries(point_cloud_objects)
                    o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
                    files.download("apoint_cloud.ply")

            color = (0, 255, 0) if label in vehicle_classes else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {real_distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Çıkış için 'q'
        break

cap.release()
cv2.destroyAllWindows()
