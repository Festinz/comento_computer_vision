# Comento Computer Vision 
컴퓨터 비전 프로젝트 - 이미지 처리 및 전처리

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
