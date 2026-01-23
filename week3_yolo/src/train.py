"""
Week3: YOLOv8을 활용한 객체 탐지 모델 학습
업무 안내서 [수행 방법] 2번 기반

실행: python train.py
"""

from ultralytics import YOLO

# ============================================
# 1. 기본 모델 학습
# ============================================

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")  # YOLOv8 기본 모델 사용

# 사용자 데이터셋으로 학습 (data.yaml 파일 필요)
model.train(data="data.yaml", epochs=10, imgsz=640)


# ============================================
# 2. 데이터 증강 적용 후 학습 (성능 향상)
# ============================================

# 데이터 증강(Augmentation): 이미지 회전, 밝기 조절, 노이즈 추가
# Hyperparameter Tuning: 학습률 조정, Batch Size 조정
# 더 깊은 모델 사용: yolov8s.pt, yolov8m.pt 등 더 큰 모델 활용

# 데이터 증강 적용 후 학습
# model.train(data="data.yaml", epochs=20, imgsz=640, augment=True)
# → 20 Epoch 동안 증강된 데이터로 학습 진행
