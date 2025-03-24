import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_results(image_name, mean_L, mean_A, mean_B, percentages, output_dir="results"):
    """ บันทึกผลการวิเคราะห์ลงไฟล์ CSV """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_csv = os.path.join(output_dir, "soil_analysis.csv")

    # เก็บข้อมูลสีเฉลี่ย
    df_color = pd.DataFrame({
        "Image": [image_name],
        "L": [mean_L], "A": [mean_A], "B": [mean_B]
    })

    # เก็บข้อมูลเปอร์เซ็นต์ของคลัสเตอร์สี
    df_percentage = pd.DataFrame(list(percentages.items()), columns=["Cluster", "Percentage"])
    df_percentage["Image"] = image_name

    df_color.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
    df_percentage.to_csv(output_csv.replace(".csv", "_clusters.csv"), mode='a', header=not os.path.exists(output_csv.replace(".csv", "_clusters.csv")), index=False)

    print(f"✅ บันทึกผลการวิเคราะห์ลงไฟล์: {output_csv}")

def plot_results(original, segmented_image, percentages):
    """ แสดงผลภาพที่ผ่านการประมวลผล """
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("K-Means Clustering")

    plt.show()

    # กำหนดสีของแต่ละคลัสเตอร์แบบสุ่ม
    colors = np.random.rand(len(percentages), 3)

    # แสดงเปอร์เซ็นต์ของแต่ละสีในกราฟแท่ง
    plt.bar(percentages.keys(), percentages.values(), color=colors)
    plt.xlabel("Color Clusters")
    plt.ylabel("Percentage (%)")
    plt.title("Percentage of Soil Color Clusters")
    plt.show()
