import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from scipy.stats import mode
from sklearn.model_selection import train_test_split

# ฟังก์ชันแปลง RGB → HSI
def rgb_to_hsi(image):
    image = image.astype(np.float32) / 255.0
    r, g, b = cv2.split(image)
    I = (r + g + b) / 3
    min_rgb = np.minimum(np.minimum(r, g), b)
    S = 1 - (min_rgb / (I + 1e-6))
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(num / den)
    H = np.where(b > g, 2 * np.pi - theta, theta) / (2 * np.pi)
    return cv2.merge([H, S, I])

# ฟังก์ชันโหลดและแปลงภาพ
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi_image = rgb_to_hsi(image)
    return image, hsi_image

# ฟังก์ชันดึงข้อมูลพิกเซล
def extract_features(image, mask, label):
    h, s, i = cv2.split(image)
    pixels = np.column_stack((h[mask > 0], s[mask > 0], i[mask > 0]))
    labels = np.full(len(pixels), label)
    return pd.DataFrame(pixels, columns=['H', 'S', 'I']).assign(Label=labels)

# ฟังก์ชันคำนวณ Adjusted Rand Index (ARI)
def kmeans_ari(X, y_true, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    return adjusted_rand_score(y_true, y_pred)

# ฟังก์ชันคำนวณ Purity Score
def purity_score(y_true, y_pred):
    """ คำนวณ Purity Score ของ K-Means """
    clusters = np.unique(y_pred)
    total_correct = 0

    for cluster in clusters:
        mask = (y_pred == cluster)
        true_labels_in_cluster = pd.Series(y_true[mask])  # ใช้ Pandas Series
        most_common_label = true_labels_in_cluster.mode()[0]  # หา Label ที่พบบ่อยสุด
        total_correct += np.sum(true_labels_in_cluster == most_common_label)

    purity = total_correct / len(y_true)
    print(f"K-Means Purity Score: {purity:.2f}")
    return purity


# ฟังก์ชันวัด Silhouette Score
def silhouette_kmeans(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    return silhouette_score(X, labels)

# ฟังก์ชันฝึก Supervised Learning Model
def train_supervised_model(model, X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    return model, accuracy

# โหลดภาพและสร้าง Dataset
BASE_PATH = os.path.normpath("C:/Users/HP/Documents/GitHub/-Soil-Organic-Matter-Estimation/dataset/Soil types")
SOIL_TYPE = "Yellow Soil"
IMAGE_NAME = "24.jpg"
IMAGE_PATH = os.path.join(BASE_PATH, SOIL_TYPE, IMAGE_NAME)

original, hsi_image = load_and_convert_image(IMAGE_PATH)

# ดึงข้อมูลพิกเซลมาใช้ฝึกโมเดล
df_dark = extract_features(hsi_image, mask=np.ones_like(hsi_image[:, :, 0]), label="Dark Soil")
df_light = extract_features(hsi_image, mask=np.ones_like(hsi_image[:, :, 0]), label="Light Soil")
df = pd.concat([df_dark, df_light])
X, y = df[['H', 'S', 'I']], df['Label']

# เปรียบเทียบโมเดล
results = {}

# K-Means Clustering
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)
results["K-Means ARI"] = kmeans_ari(X, y, k=2)
results["K-Means Purity"] = purity_score(y, y_kmeans)
results["K-Means Silhouette"] = silhouette_kmeans(X, k=2)

# Random Forest
rf_model, rf_acc = train_supervised_model(RandomForestClassifier(n_estimators=100), X, y, "Random Forest")
results["Random Forest Accuracy"] = rf_acc

# SVM
svm_model, svm_acc = train_supervised_model(SVC(kernel='rbf', C=1.0, gamma='scale'), X, y, "SVM")
results["SVM Accuracy"] = svm_acc

# KNN
knn_model, knn_acc = train_supervised_model(KNeighborsClassifier(n_neighbors=5), X, y, "KNN")
results["KNN Accuracy"] = knn_acc

# แสดงผลการเปรียบเทียบ
print("\nเปรียบเทียบผลลัพธ์ของแต่ละโมเดล:")
for key, value in results.items():
    print(f"{key}: {value:.2f}")
