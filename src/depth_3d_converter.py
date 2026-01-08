"""
2D 이미지를 3D로 변환하는 알고리즘 모듈
- Depth Map 생성
- 3D Point Cloud 변환
- 시각화 기능

Author: Shin
Date: 2025-01-09
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


def generate_depth_map(image: np.ndarray, method: str = "gradient") -> np.ndarray:
    """
    2D 이미지에서 가상의 Depth Map을 생성합니다.
    
    Args:
        image: 입력 이미지 (BGR 형식)
        method: 깊이 추정 방법 ("gradient", "intensity", "edge")
    
    Returns:
        depth_map: 깊이 맵 (grayscale)
    
    Raises:
        ValueError: 입력 이미지가 None이거나 유효하지 않은 경우
        TypeError: 입력이 numpy array가 아닌 경우
    """
    # 입력 검증
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    
    if not isinstance(image, np.ndarray):
        raise TypeError("입력은 numpy ndarray 형식이어야 합니다.")
    
    if image.size == 0:
        raise ValueError("빈 이미지입니다.")
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()
    
    # 방법에 따른 깊이 맵 생성
    if method == "gradient":
        # Sobel 기반 기울기 깊이 추정
        sobelx = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
    
    elif method == "intensity":
        # 밝기 기반 깊이 추정 (밝은 부분 = 가까움)
        depth_map = grayscale.copy()
    
    elif method == "edge":
        # 엣지 기반 깊이 추정
        edges = cv2.Canny(grayscale, 50, 150)
        depth_map = cv2.GaussianBlur(edges, (15, 15), 0)
    
    else:
        raise ValueError(f"지원하지 않는 방법입니다: {method}")
    
    return depth_map


def apply_colormap(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Depth Map에 컬러맵을 적용하여 시각화합니다.
    
    Args:
        depth_map: 깊이 맵 (grayscale)
        colormap: OpenCV 컬러맵 상수
    
    Returns:
        colored_depth: 컬러맵이 적용된 깊이 맵
    """
    if depth_map is None:
        raise ValueError("입력된 깊이 맵이 없습니다.")
    
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("입력은 numpy ndarray 형식이어야 합니다.")
    
    # 2D 배열인지 확인
    if len(depth_map.shape) != 2:
        raise ValueError("깊이 맵은 2D 배열이어야 합니다.")
    
    colored_depth = cv2.applyColorMap(depth_map, colormap)
    return colored_depth


def convert_to_3d_points(depth_map: np.ndarray, 
                         scale_z: float = 1.0,
                         downsample: int = 1) -> np.ndarray:
    """
    Depth Map을 기반으로 3D 포인트 클라우드를 생성합니다.
    
    Args:
        depth_map: 깊이 맵 (grayscale)
        scale_z: Z축 스케일 팩터
        downsample: 다운샘플링 비율 (성능 최적화)
    
    Returns:
        points_3d: 3D 좌표 배열 (N, 3) - [X, Y, Z]
    """
    if depth_map is None:
        raise ValueError("입력된 깊이 맵이 없습니다.")
    
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("입력은 numpy ndarray 형식이어야 합니다.")
    
    if scale_z <= 0:
        raise ValueError("scale_z는 양수여야 합니다.")
    
    if downsample < 1:
        raise ValueError("downsample은 1 이상이어야 합니다.")
    
    # 다운샘플링 적용
    if downsample > 1:
        depth_map = depth_map[::downsample, ::downsample]
    
    h, w = depth_map.shape[:2]
    
    # 메쉬그리드 생성
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Z값은 깊이 맵에서 가져옴
    Z = depth_map.astype(np.float32) * scale_z
    
    # 3D 포인트 배열 생성 (N, 3)
    points_3d = np.dstack((X, Y, Z)).reshape(-1, 3)
    
    return points_3d


def save_point_cloud_ply(points_3d: np.ndarray, 
                         colors: Optional[np.ndarray], 
                         filename: str) -> bool:
    """
    3D 포인트 클라우드를 PLY 파일로 저장합니다.
    
    Args:
        points_3d: 3D 좌표 배열 (N, 3)
        colors: 색상 배열 (N, 3) RGB, 선택사항
        filename: 저장할 파일 경로
    
    Returns:
        성공 여부
    """
    if points_3d is None or len(points_3d) == 0:
        raise ValueError("포인트 클라우드가 비어있습니다.")
    
    num_points = len(points_3d)
    
    with open(filename, 'w') as f:
        # PLY 헤더
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # 포인트 데이터 작성
        for i in range(num_points):
            x, y, z = points_3d[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")
    
    return True


def process_2d_to_3d(image_path: str, 
                     output_dir: str = "./output",
                     depth_method: str = "gradient") -> dict:
    """
    전체 2D → 3D 변환 파이프라인을 실행합니다.
    
    Args:
        image_path: 입력 이미지 경로
        output_dir: 출력 디렉토리
        depth_method: 깊이 추정 방법
    
    Returns:
        결과 정보 딕셔너리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    # Depth Map 생성
    depth_map = generate_depth_map(image, method=depth_method)
    
    # 컬러맵 적용
    colored_depth = apply_colormap(depth_map)
    
    # 3D 포인트 클라우드 생성
    points_3d = convert_to_3d_points(depth_map, scale_z=0.5, downsample=2)
    
    # 결과 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    depth_path = os.path.join(output_dir, f"{base_name}_depth.png")
    colored_path = os.path.join(output_dir, f"{base_name}_depth_colored.png")
    ply_path = os.path.join(output_dir, f"{base_name}_points.ply")
    
    cv2.imwrite(depth_path, depth_map)
    cv2.imwrite(colored_path, colored_depth)
    
    # PLY 저장 (색상 포함)
    h, w = depth_map.shape[:2]
    downsampled_image = image[::2, ::2]
    colors = downsampled_image.reshape(-1, 3)[:, ::-1]  # BGR to RGB
    save_point_cloud_ply(points_3d, colors, ply_path)
    
    return {
        "original_shape": image.shape,
        "depth_map_shape": depth_map.shape,
        "num_3d_points": len(points_3d),
        "depth_path": depth_path,
        "colored_path": colored_path,
        "ply_path": ply_path
    }


if __name__ == "__main__":
    # 테스트용 샘플 이미지 생성
    sample_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # 그라데이션 패턴 추가
    for i in range(200):
        sample_image[:, i] = [i, 255-i, 128]
    
    # 원 추가
    cv2.circle(sample_image, (100, 100), 50, (255, 255, 255), -1)
    
    # 저장
    cv2.imwrite("sample_input.jpg", sample_image)
    
    # 파이프라인 실행
    result = process_2d_to_3d("sample_input.jpg")
    print("처리 완료!")
    print(f"원본 이미지 크기: {result['original_shape']}")
    print(f"깊이 맵 크기: {result['depth_map_shape']}")
    print(f"3D 포인트 수: {result['num_3d_points']}")
