import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import load_and_convert_image, remove_background, analyze_soil_color
from clustering import kmeans_clustering, calculate_color_percentage, classify_soil_intensity, classify_clusters
from utils import plot_histogram

# กำหนดพาธรูปภาพ
BASE_PATH = os.path.normpath("C:/Users/HP/Documents/GitHub/-Soil-Organic-Matter-Estimation/dataset/Soil types")
SOIL_TYPE = "Yellow Soil"
IMAGE_NAME = "2.jpg"
IMAGE_PATH = os.path.join(BASE_PATH, SOIL_TYPE, IMAGE_NAME)

# วิเคราะห์ภาพดิน
original, hsi_image = load_and_convert_image(IMAGE_PATH)
clean_image, mask = remove_background(original)
mean_H, mean_S, mean_I = analyze_soil_color(hsi_image, mask)
segmented_image, labels, centers = kmeans_clustering(hsi_image, k=3)
percentages = calculate_color_percentage(labels, k=3)
dark_soil_ratio, light_soil_ratio = classify_soil_intensity(hsi_image)
cluster_analysis = classify_clusters(centers)

# แสดงผล
print(f"\n Analyzing image: {IMAGE_PATH}\n")
print(f" Average Soil Color Values:\nH = {mean_H:.2f}, S = {mean_S:.2f}, I = {mean_I:.2f}\n")

print(" Soil Color Percentages:")
for cluster, percent in percentages.items():
    print(f"{cluster}: {percent:.1f}%")

print("\n Cluster Classification:")
for cluster_info in cluster_analysis:
    print(cluster_info)

print(f"\n Soil Intensity Analysis:")
print(f" Dark Soil: {dark_soil_ratio:.2f}% (High Organic Matter)")
print(f" Light Soil: {light_soil_ratio:.2f}% (Low Organic Matter)")

# แสดงภาพ + Histogram
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title("K-Means Clustering")
plt.show()

plot_histogram(hsi_image)
