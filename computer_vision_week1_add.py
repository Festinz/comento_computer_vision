import os
from PIL import Image
import numpy as np
from datasets import load_dataset

def detect_outliers(img_array):
    """
    이상치 탐지 함수
    - 너무 어두운 이미지 필터링
    - 객체 크기가 너무 작은 이미지 필터링
    """
    # 1. 평균 밝기 체크 (너무 어두운 이미지)
    mean_brightness = np.mean(img_array)
    if mean_brightness < 50:  # 임계값: 50
        return False, f"너무 어두운 이미지 (평균 밝기: {mean_brightness:.2f})"
    
    # 2. 객체 크기 체크 (간단한 방법: 픽셀 분산)
    pixel_variance = np.var(img_array)
    if pixel_variance < 100:  # 임계값: 100
        return False, f"객체가 거의 없거나 너무 작음 (분산: {pixel_variance:.2f})"
    
    return True, "정상 이미지"

def preprocess_image(img, output_dir, image_name):
    """
    이미지 전처리 함수
    - 이상치 탐지
    - 크기 조정 (224x224)
    - 색상 변환 (Grayscale & Normalize)
    - 노이즈 제거 (Blur 필터)
    - 데이터 증강 (좌우 반전, 회전, 색상 변화)
    """
    # PIL Image로 변환 (Hugging Face 데이터셋은 PIL Image 형태)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # 이상치 탐지
    img_gray_array = np.array(img.convert('L'))
    is_valid, message = detect_outliers(img_gray_array)
    print(f"\n이상치 탐지 결과 ({image_name}): {message}")
    
    if not is_valid:
        print(f"⚠️  이미지 전처리 건너뜀: {image_name}")
        return False
    
    # 1. 크기 조정 (224x224)
    img_resized = img.resize((224, 224))
    
    # 2. Grayscale 변환 및 Normalize
    img_gray = img_resized.convert('L')
    img_gray_normalized = np.array(img_gray) / 255.0
    
    # 3. Blur 필터 적용 (노이즈 제거)
    from PIL import ImageFilter
    img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
    
    # 4. 데이터 증강
    # 좌우 반전
    img_flipped = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 회전 (15도)
    img_rotated = img_resized.rotate(15)
    
    # 색상 변화 (밝기 조정)
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(img_resized)
    img_brightened = enhancer.enhance(1.3)  # 30% 밝게
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = image_name.replace('.jpg', '')
    
    img_resized.save(os.path.join(output_dir, f"{base_name}_resized.jpg"))
    Image.fromarray((img_gray_normalized * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{base_name}_gray_normalized.jpg")
    )
    img_blurred.save(os.path.join(output_dir, f"{base_name}_blurred.jpg"))
    img_flipped.save(os.path.join(output_dir, f"{base_name}_flipped.jpg"))
    img_rotated.save(os.path.join(output_dir, f"{base_name}_rotated.jpg"))
    img_brightened.save(os.path.join(output_dir, f"{base_name}_brightened.jpg"))
    
    print(f"\n✅ 전처리 완료: {image_name}")
    print(f"저장 위치: {output_dir}")
    
    return True

if __name__ == "__main__":
    print("Hugging Face food101 데이터셋 로딩 중...")
    
    # Hugging Face에서 food101 데이터셋 로드
    # 전체 데이터셋은 크므로 일부만 사용 (train split의 일부)
    dataset = load_dataset("ethz/food101", split="train[:5]")
    
    # 출력 디렉토리
    output_directory = "preprocessed_samples"
    
    print(f"\n총 {len(dataset)}개의 이미지를 처리합니다.\n")
    
    success_count = 0
    fail_count = 0
    
    # 데이터셋의 각 이미지 처리
    for idx, item in enumerate(dataset):
        image = item['image']  # PIL Image
        image_name = f"food101_image_{idx}"
        
        success = preprocess_image(image, output_directory, image_name)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*50)
    print(f"🎉 전처리 완료!")
    print(f"✅ 성공: {success_count}개")
    print(f"⚠️  실패 (이상치): {fail_count}개")
    print("="*50)