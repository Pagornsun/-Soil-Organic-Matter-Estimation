import cv2
import numpy as np

def kmeans_clustering(hsi_image, k=3):
    """
    ใช้ K-Means Clustering เพื่อแบ่งพิกเซลในภาพ HSI ออกเป็น k กลุ่ม
    ขั้นตอนและรายละเอียด:
      1. แปลงภาพ HSI ให้อยู่ในรูปแบบ 2 มิติ (n_pixels x 3) โดยที่แต่ละแถวคือค่าของ (H, S, I)
      2. เปลี่ยนข้อมูลเป็น float32 เนื่องจาก cv2.kmeans ต้องการข้อมูลชนิดนี้
      3. กำหนดเกณฑ์หยุด (criteria) ซึ่งจะหยุดเมื่อจำนวน iteration ถึง 10 หรือเมื่อการเปลี่ยนแปลงเล็กน้อยน้อยกว่า 1.0
      4. เรียกใช้ cv2.kmeans โดยใช้ flag cv2.KMEANS_RANDOM_CENTERS เพื่อสุ่มค่าเริ่มต้นของศูนย์กลางแต่ละคลัสเตอร์
      5. เมื่อได้ผลลัพธ์จาก cv2.kmeans จะได้:
            - labels: อาร์เรย์ที่บอกว่าพิกเซลแต่ละจุดอยู่ในคลัสเตอร์ไหน
            - centers: ค่าศูนย์กลางของแต่ละคลัสเตอร์ (ในรูปแบบ H, S, I)
      6. นำค่า labels มา map แต่ละพิกเซลให้เป็นค่าใน centers เพื่อสร้างภาพที่แบ่งกลุ่ม (segmented_image)
    
    หมายเหตุ:
      - เนื่องจากภาพ HSI ที่เราใช้มีช่วงค่าอยู่ใน [0, 1] การแสดงผล segmented_image อาจต้องปรับขนาดหรือแปลงกลับ (ขึ้นอยู่กับการใช้งาน)
    
    คืนค่า:
      segmented_image: ภาพที่แบ่งกลุ่มด้วยค่าเฉลี่ยของแต่ละคลัสเตอร์
      labels: อาร์เรย์ที่บอกพิกเซลแต่ละจุดอยู่ในกลุ่มใด
      centers: ค่าศูนย์กลางของแต่ละคลัสเตอร์
    """
    # 1. แปลงภาพ HSI ให้เป็นข้อมูล 2 มิติ (n_pixels x 3)
    pixels = hsi_image.reshape((-1, 3))
    # 2. เปลี่ยนข้อมูลเป็น float32
    pixels = np.float32(pixels)
    
    # 3. กำหนดเกณฑ์หยุด (criteria) เมื่อถึง 10 รอบ หรือการเปลี่ยนแปลงเล็กน้อยน้อยกว่า 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # 4. เรียกใช้ cv2.kmeans เพื่อแบ่งพิกเซลออกเป็น k กลุ่ม
    #    cv2.KMEANS_RANDOM_CENTERS ใช้สำหรับสุ่มจุดเริ่มต้นของคลัสเตอร์
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 5. สร้าง segmented image โดย map แต่ละพิกเซลให้เป็นค่าเฉลี่ยของคลัสเตอร์ที่มันถูกจัดอยู่
    segmented_pixels = centers[labels.flatten()]
    segmented_image = segmented_pixels.reshape(hsi_image.shape)
    
    # หมายเหตุ: centers และ segmented_image ยังคงอยู่ในช่วง [0, 1]
    return segmented_image, labels, centers


def calculate_color_percentage(labels, k):
    """คำนวณเปอร์เซ็นต์ของแต่ละกลุ่มสีดิน"""
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = np.sum(counts)
    percentages = {f'Cluster {i}': (counts[i] / total_pixels) * 100 for i in range(k)}
    return percentages

def classify_soil_intensity(hsi_image, threshold=0.5):
    """แยกดินเข้ม/อ่อน โดยใช้ค่า Intensity"""
    _, _, intensity = cv2.split(hsi_image)
    dark_soil_ratio = np.sum(intensity < threshold) / intensity.size * 100
    light_soil_ratio = 100 - dark_soil_ratio
    return dark_soil_ratio, light_soil_ratio

def classify_clusters(centers):
    """จำแนกดินแต่ละกลุ่มว่าเป็นดินเข้มหรือดินอ่อน"""
    cluster_types = []
    for i, center in enumerate(centers):
        H, S, I = center
        soil_type = "Dark Soil (High Organic Matter)" if I <= 0.5 else "Light Soil (Low Organic Matter)"
        cluster_types.append(f"Cluster {i}: {soil_type} (I = {I:.2f})")
    return cluster_types
