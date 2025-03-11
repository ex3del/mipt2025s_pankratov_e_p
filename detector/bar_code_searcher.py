import cv2
import torch
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

# Инициализация клиента
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # URL API Roboflow
    api_key="08MtOSIgSAYSJMveIpLv"         # API-ключ
)

RESULTS_DIR = "results"  
os.makedirs(RESULTS_DIR, exist_ok=True)  

def detect_barcodes(image_path):
    """Выполняет инференс на изображении и сохраняет результат."""
    result = CLIENT.infer(image_path, model_id="barcode-detection-mziov/2")

    # Загрузка изображения
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Отрисовка bounding boxes и меток
    for prediction in result["predictions"]:
        x = int(prediction["x"])
        y = int(prediction["y"])
        width = int(prediction["width"])
        height = int(prediction["height"])
        confidence = prediction["confidence"]
        class_name = prediction["class"]

        # Координаты bounding box
        x1 = x - width // 2
        y1 = y - height // 2
        x2 = x + width // 2
        y2 = y + height // 2

        # Рисуем bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Result saved to {result_path}")

if __name__ == "__main__":
    input_image = "/app/data/test_image.jpg" 
    detect_barcodes(input_image)