# Araç Yakınlık Tespiti ve 3D Modelleme Sistemi

## **🔍 Projenin Vizyonu ve Amacı**
Bu proje, eski model araçların yalnızca bir telefon kamerasını kullanarak kazaları önleyebileceği bir sistem sunmaktadır. Amacımız, **LIDAR kameralarının karmaşık donanım gereksinimlerini ortadan kaldırmak** ve standart bir telefon kamerasıyla aynı işi gerçekleştirmektir. 

Projenin odak noktası, araçların mesafesini **gerçek zamanlı olarak tespit etmek** ve sürücüye erken uyarılar sağlamaktır. **Kaza riski oluştuğunda, sistem anında uyarı verir, ekran görüntüsü alır ve kaza anının 3D modelini oluşturur**. Bu 3D model, kaza anını simüle ederek, kimin suçlu kimin suçsuz olduğunu belirlemek için **kanıt niteliğinde bir görselleştirme sunar**. Bu sistem, hem sürücü destek sistemleri hem de kaza sonrası analiz için eşsiz bir araçtır.

**print('ALARM') satırı, aslında sürücü için ani bir fren sinyalidir**. Bu uyarı, bir çarpışma olasılığı olduğunda sürücünün zamanında tepki verebilmesi için bir uyarı mekanizmasıdır.

---

## **🛠️ Kullanılan Teknolojiler ve Kütüphaneler**

- **OpenCV**: Video akışından kareler almak ve işlemek için kullanılır.
- **YOLOv8**: Araç (car, truck, bus) ve diğer nesnelerin (insan, köpek, ağaç) tespiti için kullanılır.
- **MiDaS**: Derinlik tahmini sağlar, her karedeki nesneler için derinlik haritası çıkarır.
- **Open3D**: 3D modelleme ve görselleştirme için kullanılır. 3D **küre** ve **küp** nesneleri oluşturur.
- **Google Colab**: Video akışını görüntülemek ve ekran görüntülerini göstermek için `cv2_imshow` kullanılır.

---

## **⚙️ Nasıl Çalışır?**

1. **Video Girişi**: Kullanıcı kamerasını açar.
2. **Nesne Tespiti**: YOLOv8 modeli, her karede **araçları (car, truck, bus)** ve diğer nesneleri (insan, köpek, ağaç) tespit eder.
3. **Derinlik Tahmini**: MiDaS ile her nesnenin derinlik haritası çıkarılır.
4. **Mesafe Hesaplama**: İlk tespit edilen araçtan alınan derinlik verisiyle bir **ölçek faktörü** hesaplanır. Bu ölçek faktörü, diğer nesnelerin mesafesini doğru bir şekilde tahmin etmek için kullanılır.
5. **ALARM Sistemi**: Bir araç **1 metreden daha yakına** geldiğinde:
   - **ALARM verilir** (ani fren uyarısı olarak algılanır)
   - **Ekran görüntüsü alınır ve kaydedilir**
   - **3D model oluşturulur ve Open3D ile görselleştirilir**
6. **3D Modelleme**:
   - **Araçlar (car, truck, bus) küre olarak modellenir**
   - **Diğer nesneler (insan, köpek, ağaç) küp olarak modellenir**

---

## **✨ Özellikler**

1. **Gerçek Zamanlı Nesne Takibi**:
   - Video kareleri sürekli analiz edilir.
   - **Araba, Kamyon, Otobüs** gibi araçlar takip edilir.

2. **Derinlik Tahmini ve Mesafe Ölçümü**:
   - Her nesne için derinlik tahmini yapılır.
   - Derinlik değerine göre mesafe hesaplanır.
   - **Her karede ölçek faktörü güncellenir**, böylece mesafe doğru bir şekilde tahmin edilir.

3. **ALARM Sistemi**:
   - Araç **1 metreden daha yakına** geldiğinde, ekrana "**ALARM**" yazdırılır.
   - **Ekran görüntüsü alınır ve proje dizininde kaydedilir**.
   - **3D Nokta Bulutu Modeli** oluşturulur ve Open3D'de görselleştirilir.

4. **3D Modelleme**:
   - **Araçlar (car, truck, bus)** küre olarak 3D modelde gösterilir.
   - **Diğer nesneler (insan, köpek, ağaç) küp olarak gösterilir.**

---

## **📦 Kurulum**

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install opencv-python-headless torch torchvision ultralytics open3d pillow
   ```
2. Video dosyasını ekleyin ve video yolunu ayarlayın:
   ```python
   video_path = 'videoarabasurus.mp4'  # Buraya video dosyasının yolu yazılmalı
   ```

---

## **🧱 Kodun Ana Parçaları**

### **1️⃣ Video İşleme**
```python
cap = cv2.VideoCapture(0)  #  Kamerayı aç
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
```
Bu bölümde video kare kare işlenir.

---

### **2️⃣ YOLO Nesne Tespiti**
```python
results = model(frame)  # YOLOv8 ile nesne tespiti
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
```
Bu kısım YOLOv8 modelini kullanarak her karedeki nesneleri tespit eder.

---

### **3️⃣ Derinlik Tahmini ve Mesafe Hesaplama**
```python
with torch.no_grad():
    depth_map = midas(input_batch)
    avg_depth = np.mean(depth_map)
    if label in vehicle_classes:
        reference_avg_depth = avg_depth
        scale_factor = reference_distance_m / reference_avg_depth
real_distance = avg_depth * scale_factor
```
Bu bölümde MiDaS kullanılarak derinlik tahmini yapılır ve mesafe hesaplanır.

---

### **4️⃣ ALARM Sistemi ve 3D Modelleme**
```python
if real_distance < 1.0 and label in vehicle_classes:
    print(f"ALARM: {label} 1 metre altına yaklaştı!")
    ss_path = f"screenshot_{time.strftime('%Y%m%d-%H%M%S')}.png"
    cv2.imwrite(ss_path, frame)
```
Bu bölümde bir araç 1 metreden daha yakına gelirse alarm verilir, ekran görüntüsü alınır ve kaydedilir.

---

## **🔚 Sonuç**
Bu proje, **araçların (car, truck, bus) mesafesini gerçek zamanlı olarak izlemek** ve **1 metreden yakına geldiklerinde alarm vermek** için kullanılır. Bu tür bir sistem, **sürücü asistan sistemleri** ve **kaza sonrası analiz sistemleri** için faydalı olabilir.

---

## **💡 Geliştirme Fikirleri**
1. **Geliştirilmiş Kalibrasyon**: Daha doğru bir ölçek faktörü hesaplamak için birden fazla araçtan veri alınabilir.
2. **Daha Fazla Nesne Desteği**: İnsan, bisiklet, motosiklet gibi ek nesneler izlenebilir.
3. **Mesafe Görselleştirmesi**: Görüntüde mesafeyi göstermek için renk kodlaması eklenebilir.

## ÖNEMLİ NOT

Bu kod, 3D modelleme içerdiğinden oldukça güçlü bir sistem gerektirmektedir. 
