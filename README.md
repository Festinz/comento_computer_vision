# Comento Computer Vision

컴퓨터 비전 프로젝트 - 이미지 처리, 전처리 및 2D→3D 변환

---

# 📌 Week 1: 이미지 처리 및 전처리

## 프로젝트 구조

```
comento_computer_vision/
├── computer_vision_week1_base.py    # 빨간색 검출 코드
├── computer_vision_week1_add.py     # 이미지 전처리 코드 (Hugging Face 데이터셋 사용)
├── sample.jpg                        # 테스트 이미지
├── preprocessed_samples/             # 전처리된 이미지 저장 폴더
└── README.md
```

## 기능

### 1. 빨간색 검출 (computer_vision_week1_base.py)
- OpenCV를 사용한 HSV 색상 공간 기반 빨간색 영역 검출
- 두 개의 빨간색 범위를 설정하여 정확한 검출

### 2. 이미지 전처리 (computer_vision_week1_add.py)

#### 데이터셋
- **Hugging Face food101 데이터셋** 사용
- URL: https://huggingface.co/datasets/ethz/food101
- 5개 샘플 이미지로 테스트

#### 이상치 탐지
- **너무 어두운 이미지 필터링**: 평균 밝기가 50 미만인 이미지 제거
- **객체 크기 검증**: 픽셀 분산이 100 미만인 이미지 제거

#### 전처리 과정
1. **크기 조정**: 모든 이미지를 224x224 크기로 통일
2. **색상 변환**: Grayscale 변환 및 0-1 사이로 정규화
3. **노이즈 제거**: Gaussian Blur 필터 적용 (radius=2)
4. **데이터 증강**:
   - 좌우 반전
   - 15도 회전
   - 밝기 조정 (30% 증가)

## 사용법

### 빨간색 검출
```bash
python computer_vision_week1_base.py
```

### 이미지 전처리 (Hugging Face 데이터셋)
```bash
python computer_vision_week1_add.py
```

## 필요한 패키지
```bash
pip install opencv-python numpy pillow datasets huggingface-hub
```

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

## 📚 참고 자료

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [PLY File Format](http://paulbourke.net/dataformats/ply/)

## 👤 Author

- **Shin** - 의공학/전기전자공학 전공
- Date: 2025-01-09
