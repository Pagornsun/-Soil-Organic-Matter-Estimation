import os
from image_processing import load_and_convert_image, remove_background, analyze_soil_color
from clustering import kmeans_clustering, calculate_color_percentage
from utils import save_results, plot_results

if __name__ == "__main__":
    # กำหนดเส้นทางไฟล์รูปภาพ
    BASE_PATH = "dataset/Soil types"
    SOIL_TYPE = "Cinder Soil"
    IMAGE_NAME = "23.jpg"
    IMAGE_PATH = os.path.join(BASE_PATH, SOIL_TYPE, IMAGE_NAME)

    # ตรวจสอบว่าไฟล์มีอยู่จริง
    if not os.path.exists(IMAGE_PATH):
        print(f"File not found: {IMAGE_PATH}")
    else:
        print(f"Analyzing image: {IMAGE_PATH}")

        original, hsi_image = load_and_convert_image(IMAGE_PATH)

        clean_image, mask = remove_background(original)

        mean_H, mean_S, mean_I = analyze_soil_color(hsi_image, mask)
        print(f"Average Soil Color Values: H={mean_H:.2f}, S={mean_S:.2f}, I={mean_I:.2f}")

        K = 3
        segmented_image, labels, centers = kmeans_clustering(hsi_image, k=K)
        
        percentages = calculate_color_percentage(labels, k=K)
        print("Soil Color Percentages:", percentages)

        save_results(IMAGE_NAME, mean_H, mean_S, mean_I, percentages)

        plot_results(original, segmented_image, percentages)
