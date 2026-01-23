# Comento Computer Vision

컴퓨터 비전 프로젝트 - 이미지 처리, 전처리, 2D→3D 변환 및 객체 탐지

---

## 📁 프로젝트 구조
```
comento_computer_vision/
├── README.md
├── week1_preprocessing/          # Week1: 이미지 처리 및 전처리
│   ├── computer_vision_week1_base.py
│   └── sample.jpg
├── week2_2d_to_3d/               # Week2: Unit Test 및 2D→3D 변환
│   ├── src/
│   ├── tests/
│   ├── scripts/
│   └── results/
└── week3_yolo/                   # Week3: YOLOv8 객체 탐지
    ├── src/
    ├── results/
    └── datasets/
```

---

# 📌 Week 1: 이미지 처리 및 전처리

## 기능

### 1. 빨간색 검출 (computer_vision_week1_base.py)
- OpenCV를 사용한 HSV 색상 공간 기반 빨간색 영역 검출
- 두 개의 빨간색 범위를 설정하여 정확한 검출

### 2. 이미지 전처리 (computer_vision_week1_add.py)
- **Hugging Face food101 데이터셋** 사용
- 전처리: 크기 조정, Grayscale 변환, 노이즈 제거, 데이터 증강

## 실행 방법
```bash
cd week1_preprocessing
pip install opencv-python numpy pillow datasets huggingface-hub
python computer_vision_week1_base.py
```

---

# 📌 Week 2: Unit Test 구성 및 2D → 3D 변환

## 기능
- Python pytest를 활용한 Unit Test 구성
- OpenCV와 NumPy를 사용한 2D → 3D 변환 알고리즘 구현
- 깊이 맵(Depth Map) 생성 및 3D 포인트 클라우드 변환

## 주요 함수

| 함수 | 설명 |
|------|------|
| `generate_depth_map()` | 2D 이미지에서 깊이 맵 생성 |
| `apply_colormap()` | 깊이 맵에 컬러맵 적용 |
| `convert_to_3d_points()` | 깊이 맵을 3D 포인트 클라우드로 변환 |
| `save_point_cloud_ply()` | PLY 파일로 저장 |

## 실행 방법
```bash
cd week2_2d_to_3d
pip install numpy opencv-python pytest matplotlib

# Unit Test 실행
pytest tests/test_depth_3d_converter.py -v

# 시각화 데모 실행
python scripts/visualization_demo.py
```

---

# 📌 Week 3: AI 기반 객체 탐지 및 OpenCV 시각화

## 기능
- YOLOv8 모델을 활용한 커스텀 데이터셋 학습
- OpenCV를 사용한 객체 탐지 결과 시각화
- Matplotlib을 활용한 모델 성능 평가 시각화

## 실행 방법
```bash
cd week3_yolo/src
pip install torch torchvision opencv-python matplotlib ultralytics

# 모델 학습
python train.py

# 객체 탐지
python detect.py

# 결과 시각화
python visualize.py
```

## 주요 코드

### train.py
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=10, imgsz=640)
```

## 성능 지표

| 메트릭 | 설명 |
|--------|------|
| mAP@0.5 | IoU 0.5 기준 평균 정밀도 |
| Precision | 탐지한 객체 중 정답 비율 |
| Recall | 실제 객체 중 탐지한 비율 |

---

## 📚 참고 자료

- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)