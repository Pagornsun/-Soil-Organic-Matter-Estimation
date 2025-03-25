import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_histogram(hsi_image):
    """แสดง Histogram ของค่า Intensity"""
    _, _, intensity = cv2.split(hsi_image)
    plt.figure(figsize=(8, 5))
    plt.hist(intensity.ravel(), bins=50, color='brown', alpha=0.7)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.title("Soil Intensity Distribution")
    plt.show()
