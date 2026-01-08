# 2D → 3D 변환 프로젝트

AI 기반 제품 개발을 위한 Unit Test 구성 및 2D → 3D 변환 실습 프로젝트입니다.

## 📋 프로젝트 개요

본 프로젝트는 다음 목표를 달성합니다:
1. Python의 pytest를 활용한 Unit Test 구성
2. OpenCV와 NumPy를 사용한 2D → 3D 변환 알고리즘 구현
3. 깊이 맵(Depth Map) 생성 및 3D 포인트 클라우드 변환

## 🛠 환경 설정

### 필요 라이브러리 설치

```bash
pip install numpy opencv-python pytest matplotlib
```

### 디렉토리 구조

```
2d_to_3d_project/
├── depth_3d_converter.py      # 메인 변환 모듈
├── test_depth_3d_converter.py # Unit Test 코드
├── visualization_demo.py      # 시각화 데모
├── README.md                  # 프로젝트 문서
└── output/                    # 결과물 저장 폴더
```

## 🚀 실행 방법

### 1. Unit Test 실행

```bash
# 기본 실행
pytest test_depth_3d_converter.py -v

# 상세 출력
pytest test_depth_3d_converter.py -v --tb=short

# 특정 테스트만 실행
pytest test_depth_3d_converter.py::TestGenerateDepthMap -v

# 커버리지 리포트 (pytest-cov 설치 필요)
pytest test_depth_3d_converter.py -v --cov=depth_3d_converter --cov-report=html
```

### 2. 시각화 데모 실행

```bash
python visualization_demo.py
```

### 3. 개별 이미지 처리

```python
from depth_3d_converter import process_2d_to_3d

result = process_2d_to_3d("your_image.jpg", "./output")
print(f"3D 포인트 수: {result['num_3d_points']}")
```

## 📖 주요 함수 설명

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

**Parameters:**
- `depth_map`: 입력 깊이 맵
- `colormap`: OpenCV 컬러맵 (기본값: `cv2.COLORMAP_JET`)

**Returns:**
- `colored_depth`: 컬러 이미지 (BGR)

### `convert_to_3d_points(depth_map, scale_z, downsample)`
깊이 맵을 3D 포인트 클라우드로 변환합니다.

**Parameters:**
- `depth_map`: 입력 깊이 맵
- `scale_z`: Z축 스케일 팩터
- `downsample`: 다운샘플링 비율

**Returns:**
- `points_3d`: 3D 좌표 배열 (N, 3)

### `save_point_cloud_ply(points_3d, colors, filename)`
3D 포인트 클라우드를 PLY 파일로 저장합니다.

### `process_2d_to_3d(image_path, output_dir, depth_method)`
전체 2D → 3D 변환 파이프라인을 실행합니다.

## ✅ Unit Test 구성

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
collected 35 items

test_depth_3d_converter.py::TestGenerateDepthMap::test_basic_functionality PASSED
test_depth_3d_converter.py::TestGenerateDepthMap::test_output_shape PASSED
test_depth_3d_converter.py::TestGenerateDepthMap::test_none_input_raises_error PASSED
...
========================= 35 passed in 1.23s ===========================
```

## 📊 결과물

### 생성되는 파일

1. **깊이 맵 이미지** (`*_depth.png`)
   - 그레이스케일 깊이 정보

2. **컬러 깊이 맵** (`*_depth_colored.png`)
   - JET 컬러맵 적용 시각화

3. **3D 포인트 클라우드** (`*_points.ply`)
   - MeshLab, CloudCompare 등에서 확인 가능

4. **파이프라인 비교 이미지** (`*_pipeline.png`)
   - 원본 → 깊이 맵 → 3D 변환 과정

## 📚 참고 자료

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [pytest Documentation](https://docs.pytest.org/)
- [PLY File Format](http://paulbourke.net/dataformats/ply/)

## 👤 Author

- **Shin** - 의생명공학/전기전자공학 전공
- Date: 2025-01-09
