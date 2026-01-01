import os
from PIL import Image
import numpy as np

def preprocess_image(image_path, output_dir):
    """
    이미지 전처리 함수
    - 크기 조정 (224x224)
    - 색상 변환 (Grayscale & Normalize)
    - 노이즈 제거 (Blur 필터)
    - 데이터 증강 (좌우 반전, 회전, 색상 변화)
    """
    # 이미지 로드
    img = Image.open(image_path)
    
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
    
    # 파일명 추출
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    
    img_resized.save(os.path.join(output_dir, f"{base_name}_resized.jpg"))
    Image.fromarray((img_gray_normalized * 255).astype(np.uint8)).save(
        os.path.join(output_dir, f"{base_name}_gray_normalized.jpg")
    )
    img_blurred.save(os.path.join(output_dir, f"{base_name}_blurred.jpg"))
    img_flipped.save(os.path.join(output_dir, f"{base_name}_flipped.jpg"))
    img_rotated.save(os.path.join(output_dir, f"{base_name}_rotated.jpg"))
    
    print(f"전처리 완료: {image_path}")
    print(f"저장 위치: {output_dir}")
    print(f"- {base_name}_resized.jpg")
    print(f"- {base_name}_gray_normalized.jpg")
    print(f"- {base_name}_blurred.jpg")
    print(f"- {base_name}_flipped.jpg")
    print(f"- {base_name}_rotated.jpg")

if __name__ == "__main__":
    # 입력 이미지 경로
    input_image = "sample.jpg"
    
    # 출력 디렉토리
    output_directory = "preprocessed_samples"
    
    # 전처리 실행
    if os.path.exists(input_image):
        preprocess_image(input_image, output_directory)
    else:
        print(f"이미지 파일을 찾을 수 없습니다: {input_image}")
        print("sample.jpg 파일이 같은 폴더에 있는지 확인해주세요.")