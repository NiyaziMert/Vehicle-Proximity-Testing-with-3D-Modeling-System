import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
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
    Resize((384, 384)),
    ToTensor()
])

# Kalibrasyon için referans mesafe
reference_distance_m = 2.0  # Referans mesafe (örneğin, 2 metre)
reference_avg_depth = None


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

            # Yol ve tanınmayan sınıflar dikkate alınmayacak
            if label in ignored_classes:
                continue

            # Nesne bölgesini hazırla
            roi = image_rgb[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (384, 384))
            input_batch = transform(Image.fromarray(roi_resized)).unsqueeze(0)

            # MiDaS ile derinlik tahmini
            with torch.no_grad():
                depth_map = midas(input_batch)
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False
                ).squeeze().cpu().numpy()

            avg_depth = np.mean(depth_map)

            # Sürekli Ölçek Faktörü Güncellemesi
            if label in vehicle_classes:
                reference_avg_depth = avg_depth
                scale_factor = reference_distance_m / reference_avg_depth
                print(f"Kalibrasyon Güncellendi: Ölçek Faktörü = {scale_factor:.4f}")

            real_distance = avg_depth * scale_factor
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_z = real_distance

            # 1 Metreden yakın araçlarda ekran görüntüsü alın ve 3D modelle
            if real_distance < 1.0 and label in vehicle_classes:
                print(f"ALARM: {label} 1 metre altına yaklaştı! Uzaklık: {real_distance:.2f} m")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                ss_path = f"screenshot_{timestamp}.png"
                cv2.imwrite(ss_path, frame)
                print(f"Ekran görüntüsü kaydedildi: {ss_path}")

                # 3D Modelleme
                if label in vehicle_classes:
                    sphere = create_sphere([center_x, center_y, center_z])
                    point_cloud_objects.append(sphere)
                else:
                    cube = create_cube([center_x, center_y, center_z])
                    point_cloud_objects.append(cube)
                if point_cloud_objects:
                    print("3D Model Oluşturuluyor...")
                    o3d.visualization.draw_geometries(point_cloud_objects)


            # Görüntü üzerinde kutu çiz
            color = (0, 255, 0) if label in vehicle_classes else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {real_distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    # Görüntü göster
    cv2.imshow(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Çıkış için 'q'
        break

cap.release()
cv2.destroyAllWindows()
