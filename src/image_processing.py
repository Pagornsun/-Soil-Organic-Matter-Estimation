import os
import cv2
import numpy as np

def rgb_to_hsi(image):
    """แปลงภาพจาก RGB เป็น HSI Color Space"""
    image = image.astype(np.float32) / 255.0
    r, g, b = cv2.split(image)
    
    I = (r + g + b) / 3
    min_rgb = np.minimum(np.minimum(r, g), b)
    S = 1 - (min_rgb / (I + 1e-6))  # ป้องกันหารด้วย 0
    
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(num / den)
    
    H = np.where(b > g, 2 * np.pi - theta, theta)
    H = H / (2 * np.pi)  # ปรับให้ H อยู่ในช่วง 0-1
    
    hsi_image = cv2.merge([H, S, I])
    return hsi_image

def load_and_convert_image(image_path):
    """โหลดและแปลงภาพเป็น HSI"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์: {image_path}")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi_image = rgb_to_hsi(image)
    return image, hsi_image

def remove_background(image):
    """กำจัดพื้นหลังของดินด้วย Otsu's Thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def analyze_soil_color(hsi_image, mask):
    """คำนวณค่าเฉลี่ยของ H, S, I"""
    h, s, i = cv2.split(hsi_image)
    mean_H = np.mean(h[mask > 0])
    mean_S = np.mean(s[mask > 0])
    mean_I = np.mean(i[mask > 0])
    return mean_H, mean_S, mean_I
