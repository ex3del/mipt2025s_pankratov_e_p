import cv2
import torch
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "yolov10x.pt"  # Стандартные веса YOLOv10
CONF_THRESH = 0.5  # Порог уверенности для обнаружения
IOU_THRESH = 0.5  # Порог для Non-Maximum Suppression (NMS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results"

def detect_barcodes(image_path, model_path):
    """Detect barcodes in an image using YOLOv10."""
    # Load model
    model = YOLO(model_path).to(DEVICE)
    
    # Preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Inference
    results = model.predict(
        img_rgb, 
        conf=CONF_THRESH, 
        iou=IOU_THRESH, 
        imgsz=640,  # Размер изображения для inference
        augment=True  # Аугментация для улучшения точности
    )
    
    # Post-process results
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class": model.names[int(cls_id)]
            })
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, 
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            x1, y1-10, 
            f"{detection['class']} {detection['confidence']:.2f}",
            color='lime',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.7)
        )
    
    plt.axis('off')
    
    # Save result
    result_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Result saved to {result_path}")

if __name__ == "__main__":
    # Detect barcodes in the image
    input_image = "/app/data/test_image.jpg"  # Замените на путь к вашему изображению
    detect_barcodes(input_image, MODEL_NAME)