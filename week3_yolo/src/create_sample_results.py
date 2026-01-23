"""
샘플 결과 이미지 생성 (데모용)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../results", exist_ok=True)

# 1. 샘플 탐지 결과 이미지 생성
def create_detection_result():
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    img[:150, :] = [230, 200, 180]
    img[350:, :] = [100, 100, 100]
    
    cv2.rectangle(img, (50, 100), (150, 350), (150, 140, 130), -1)
    cv2.rectangle(img, (180, 80), (280, 350), (140, 150, 160), -1)
    cv2.rectangle(img, (450, 120), (600, 350), (160, 150, 140), -1)
    
    cv2.rectangle(img, (300, 200), (360, 340), (50, 50, 150), -1)
    cv2.circle(img, (330, 180), 20, (200, 180, 170), -1)
    cv2.rectangle(img, (400, 220), (450, 340), (100, 50, 50), -1)
    cv2.circle(img, (425, 200), 18, (200, 180, 170), -1)
    
    cv2.rectangle(img, (100, 360), (250, 440), (150, 50, 50), -1)
    cv2.circle(img, (130, 440), 20, (30, 30, 30), -1)
    cv2.circle(img, (220, 440), 20, (30, 30, 30), -1)
    
    cv2.ellipse(img, (520, 400), (40, 25), 0, 0, 360, (139, 119, 101), -1)
    cv2.circle(img, (550, 385), 15, (139, 119, 101), -1)
    
    detections = [
        {"bbox": (290, 155, 370, 345), "label": "person", "conf": 0.92},
        {"bbox": (390, 178, 460, 345), "label": "person", "conf": 0.87},
        {"bbox": (95, 355, 255, 465), "label": "car", "conf": 0.95},
        {"bbox": (475, 365, 580, 430), "label": "dog", "conf": 0.78},
    ]
    
    colors = {"person": (0, 255, 0), "car": (255, 0, 0), "dog": (0, 165, 255)}
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label, conf = det["label"], det["conf"]
        color = colors.get(label, (0, 255, 0))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(img, text, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.putText(img, "YOLO Object Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.imwrite("../results/detection_result.jpg", img)
    print("detection_result.jpg 저장 완료")

# 2. 샘플 성능 그래프 생성
def create_performance_graph():
    epochs = np.arange(1, 11)
    precision = np.clip(1 - 0.7*np.exp(-0.4*epochs) + np.random.normal(0, 0.02, 10), 0, 1)
    recall = np.clip(1 - 0.8*np.exp(-0.35*epochs) + np.random.normal(0, 0.02, 10), 0, 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, precision, 'g-o', label='Precision', markersize=6)
    plt.plot(epochs, recall, 'm-s', label='Recall', markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.savefig("../results/model_performance.png", dpi=150, bbox_inches='tight')
    print("model_performance.png 저장 완료")
    plt.close()

if __name__ == "__main__":
    create_detection_result()
    create_performance_graph()
