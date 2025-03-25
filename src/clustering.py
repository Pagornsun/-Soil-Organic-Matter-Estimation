import cv2
import numpy as np

def kmeans_clustering(hsi_image, k=3):
    """ใช้ K-Means Clustering แบ่งระดับสีดินใน HSI"""
    pixels = hsi_image.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(hsi_image.shape)
    
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
