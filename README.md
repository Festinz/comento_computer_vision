# Comento Computer Vision


컴퓨터 비전 프로젝트 - 이미지 처리, 전처리, 2D→3D 변환 및 객체 탐지

---

## 📁 프로젝트 구조


컴퓨터 비전 프로젝트 - 이미지 처리, 전처리 및 2D→3D 변환

---

# 📌 Week 1: 이미지 처리 및 전처리

## 프로젝트 구조


```
comento_computer_vision/
├── README.md
├── week1_preprocessing/          # Week1: 이미지 처리 및 전처리
│   ├── computer_vision_week1_base.py
│   ├── computer_vision_week1_add.py
│   ├── sample.jpg
│   └── preprocessed_samples/
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

#### 데이터셋
- **Hugging Face food101 데이터셋** 사용
- URL: https://huggingface.co/datasets/ethz/food101

#### 전처리 과정
- 크기 조정 (224x224)
- Grayscale 변환 및 정규화
- Gaussian Blur 노이즈 제거
- 데이터 증강 (좌우 반전, 회전, 밝기 조정)

## 실행 방법
```bash
cd week1_preprocessing
pip install opencv-python numpy pillow datasets huggingface-hub
python computer_vision_week1_base.py
python computer_vision_week1_add.py
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

## 프로젝트 구조
```
week3_yolo/
├── src/
│   ├── data.yaml          # 데이터셋 설정
│   ├── train.py           # 모델 학습
│   ├── detect.py          # 객체 탐지 + OpenCV 시각화
│   └── visualize.py       # 성능 그래프
├── results/
│   ├── detection_result.jpg
│   └── model_performance.png
└── datasets/
    ├── train/{images, labels}
    ├── valid/{images, labels}
    └── test/{images, labels}
```

## 실행 방법
```bash
cd week3_yolo
pip install torch torchvision opencv-python matplotlib ultralytics

# 모델 학습
cd src
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

### detect.py
```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")
results = model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

## 성능 지표

| 메트릭 | 설명 |
|--------|------|
| mAP@0.5 | IoU 0.5 기준 평균 정밀도 |
| mAP@0.5:0.95 | IoU 0.5~0.95 기준 평균 정밀도 |
| Precision | 탐지한 객체 중 정답 비율 |
| Recall | 실제 객체 중 탐지한 비율 |

## 성능 향상 방법

1. **데이터 증강**: `augment=True` 옵션 추가
2. **하이퍼파라미터 튜닝**: 학습률, Batch Size 조정
3. **더 큰 모델 사용**: YOLOv8s, YOLOv8m, YOLOv8l

---

## 📚 참고 자료

=======
## 출력 결과
전처리된 이미지는 `preprocessed_samples/` 폴더에 저장됩니다:
- `food101_image_0_resized.jpg` - 크기 조정
- `food101_image_0_gray_normalized.jpg` - Grayscale & 정규화
- `food101_image_0_blurred.jpg` - 노이즈 제거
- `food101_image_0_flipped.jpg` - 좌우 반전
- `food101_image_0_rotated.jpg` - 회전
- `food101_image_0_brightened.jpg` - 밝기 조정

(총 5개 이미지 × 6개 변형 = 30개 파일 생성)

---

# 📌 Week 2: Unit Test 구성 및 2D → 3D 변환

AI 기반 제품 개발을 위한 Unit Test 구성 및 2D → 3D 변환 실습 프로젝트입니다.

## 프로젝트 개요

본 프로젝트는 다음 목표를 달성합니다:
1. Python의 pytest를 활용한 Unit Test 구성
2. OpenCV와 NumPy를 사용한 2D → 3D 변환 알고리즘 구현
3. 깊이 맵(Depth Map) 생성 및 3D 포인트 클라우드 변환

## 프로젝트 구조

```
week2_2d_to_3d/
├── src/                           # 소스 코드
│   └── depth_3d_converter.py
├── tests/                         # Unit Test
│   └── test_depth_3d_converter.py
├── scripts/                       # 실행 스크립트
│   └── visualization_demo.py
└── results/                       # 결과 이미지
    ├── comparison.png
    ├── shapes_pipeline.png
    └── ...
```

## 환경 설정

### 필요 라이브러리 설치
```bash
pip install numpy opencv-python pytest matplotlib
```

## 실행 방법

### 1. Unit Test 실행
```bash
# 기본 실행
pytest tests/test_depth_3d_converter.py -v

# 상세 출력
pytest tests/test_depth_3d_converter.py -v --tb=short
```

### 2. 시각화 데모 실행
```bash
python scripts/visualization_demo.py
```

### 3. 개별 이미지 처리
```python
from src.depth_3d_converter import process_2d_to_3d

result = process_2d_to_3d("your_image.jpg", "./output")
print(f"3D 포인트 수: {result['num_3d_points']}")
```

## 주요 함수 설명

### `generate_depth_map(image, method)`
2D 이미지에서 깊이 맵을 생성합니다.

**Parameters:**
- `image`: 입력 이미지 (BGR 형식 또는 그레이스케일)
- `method`: 깊이 추정 방법
  - `"gradient"`: Sobel 기반 기울기 추정
  - `"intensity"`: 밝기 기반 추정
  - `"edge"`: Canny 엣지 기반 추정

**Returns:**
- `depth_map`: 깊이 맵 (uint8, grayscale)

### `apply_colormap(depth_map, colormap)`
깊이 맵에 컬러맵을 적용하여 시각화합니다.

### `convert_to_3d_points(depth_map, scale_z, downsample)`
깊이 맵을 3D 포인트 클라우드로 변환합니다.

### `save_point_cloud_ply(points_3d, colors, filename)`
3D 포인트 클라우드를 PLY 파일로 저장합니다.

### `process_2d_to_3d(image_path, output_dir, depth_method)`
전체 2D → 3D 변환 파이프라인을 실행합니다.

## Unit Test 구성

### 테스트 클래스 구조

| 클래스 | 테스트 항목 |
|--------|------------|
| `TestGenerateDepthMap` | 깊이 맵 생성 기능, 입력 검증, 다양한 방법 테스트 |
| `TestApplyColormap` | 컬러맵 적용, 출력 형식 검증 |
| `TestConvertTo3DPoints` | 3D 포인트 변환, 파라미터 검증 |
| `TestSavePointCloudPLY` | PLY 파일 저장, 형식 검증 |
| `TestProcess2DTo3D` | 통합 파이프라인 테스트 |
| `TestEdgeCases` | 경계 조건 및 엣지 케이스 |

### 테스트 결과 예시
```
========================= test session starts ==========================
collected 43 items

test_depth_3d_converter.py::TestGenerateDepthMap::test_basic_functionality PASSED
test_depth_3d_converter.py::TestGenerateDepthMap::test_output_shape PASSED
...
========================= 43 passed in 0.92s ===========================
```

## 결과물

### 생성되는 파일
1. **깊이 맵 이미지** (`*_depth.png`) - 그레이스케일 깊이 정보
2. **컬러 깊이 맵** (`*_depth_colored.png`) - JET 컬러맵 적용 시각화
3. **3D 포인트 클라우드** (`*_points.ply`) - MeshLab, CloudCompare 등에서 확인 가능
4. **파이프라인 비교 이미지** (`*_pipeline.png`) - 원본 → 깊이 맵 → 3D 변환 과정

---

# 📌 Week 3: AI 기반 객체 탐지 및 OpenCV 시각화

YOLOv8을 활용한 객체 탐지 모델 학습 및 OpenCV를 통한 결과 시각화 프로젝트입니다.

## 프로젝트 개요

본 프로젝트는 다음 목표를 달성합니다:
1. YOLOv8 모델을 활용한 커스텀 데이터셋 학습
2. OpenCV를 사용한 객체 탐지 결과 시각화
3. Matplotlib을 활용한 모델 성능 평가 시각화

## 프로젝트 구조
```
Yolo project/
├── src/
│   ├── data.yaml          # 데이터셋 설정 파일
│   ├── train.py           # 모델 학습 스크립트
│   ├── detect.py          # 객체 탐지 + OpenCV 시각화
│   └── visualize.py       # 성능 그래프 시각화
├── results/
│   ├── detection_result.jpg      # 탐지 결과 이미지
│   └── model_performance.png     # Precision/Recall 그래프
├── docs/
│   └── README.md
└── datasets/
    ├── train/{images, labels}
    ├── valid/{images, labels}
    └── test/{images, labels}
```

## 환경 설정

### 필요 라이브러리 설치
```bash
pip install torch torchvision opencv-python matplotlib ultralytics
```

## 실행 방법

### 1. 데이터셋 준비
`datasets/` 폴더에 YOLO 형식의 데이터셋을 준비합니다.

YOLO 라벨 형식 (txt 파일):
```
# class_id x_center y_center width height (0~1 정규화)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

### 2. 모델 학습
```bash
cd "Yolo project/src"
python train.py
```

**학습 파라미터:**
- Epochs: 10 (기본) / 20 (증강 적용 시)
- Image Size: 640x640
- Model: YOLOv8n (nano)

### 3. 객체 탐지
```bash
python detect.py
```

### 4. 결과 시각화
```bash
python visualize.py
```

## 주요 코드 설명

### `train.py` - 모델 학습
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 기본 모델
model.train(data="data.yaml", epochs=10, imgsz=640)
```

### `detect.py` - 객체 탐지 및 시각화
```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")
results = model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### `visualize.py` - 성능 시각화
```python
import matplotlib.pyplot as plt

metrics = model.val()
plt.plot(metrics['precision'], label="Precision")
plt.plot(metrics['recall'], label="Recall")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.title("Model Performance")
plt.savefig("../results/model_performance.png")
```

## 실행 결과

### 모델 평가 결과

| 메트릭 | 설명 |
|--------|------|
| mAP@0.5 | IoU 0.5 기준 평균 정밀도 |
| mAP@0.5:0.95 | IoU 0.5~0.95 기준 평균 정밀도 |
| Precision | 탐지한 객체 중 정답 비율 |
| Recall | 실제 객체 중 탐지한 비율 |

### 결과 이미지

1. **detection_result.jpg** - 바운딩 박스가 표시된 탐지 결과
2. **model_performance.png** - Precision/Recall 학습 곡선

## 성능 향상 방법

1. **데이터 증강 (Augmentation)**
   - 이미지 회전, 밝기 조절, 노이즈 추가
   - `model.train(data="data.yaml", epochs=20, imgsz=640, augment=True)`

2. **하이퍼파라미터 튜닝**
   - 학습률 조정
   - Batch Size 조정

3. **더 큰 모델 사용**
   - `yolov8s.pt` (small)
   - `yolov8m.pt` (medium)
   - `yolov8l.pt` (large)

---

## 📚 참고 자료
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PLY File Format](http://paulbourke.net/dataformats/ply/)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 👤 Author

- **Shin** - 의공학/전기전자공학 전공
- Date: 2025-01-09

