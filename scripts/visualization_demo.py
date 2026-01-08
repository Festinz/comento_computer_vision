"""
2D → 3D 변환 시각화 및 데모 스크립트
결과 이미지를 생성하고 저장합니다.

Author: Shin
Date: 2025-01-09
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from depth_3d_converter import (
    generate_depth_map,
    apply_colormap,
    convert_to_3d_points,
    process_2d_to_3d
)


def create_sample_images():
    """다양한 테스트용 샘플 이미지 생성"""
    samples = {}
    
    # 1. 기본 그라데이션 이미지
    gradient = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        gradient[:, i] = [i, 255-i, 128]
    samples['gradient'] = gradient
    
    # 2. 원과 사각형이 있는 이미지
    shapes = np.zeros((200, 200, 3), dtype=np.uint8)
    shapes[:] = [50, 50, 50]  # 배경
    cv2.circle(shapes, (100, 100), 60, (255, 200, 100), -1)
    cv2.rectangle(shapes, (20, 20), (80, 80), (100, 255, 100), -1)
    cv2.rectangle(shapes, (120, 120), (180, 180), (100, 100, 255), -1)
    samples['shapes'] = shapes
    
    # 3. 깊이감을 표현한 이미지 (중앙이 밝음 = 가까움)
    depth_sim = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            dist = np.sqrt((i-100)**2 + (j-100)**2)
            intensity = max(0, 255 - int(dist * 2))
            depth_sim[i, j] = [intensity, intensity, intensity]
    samples['depth_simulation'] = depth_sim
    
    # 4. 체크보드 패턴
    checker = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(0, 200, 25):
        for j in range(0, 200, 25):
            if (i // 25 + j // 25) % 2 == 0:
                checker[i:i+25, j:j+25] = [200, 200, 200]
            else:
                checker[i:i+25, j:j+25] = [50, 50, 50]
    samples['checkerboard'] = checker
    
    return samples


def visualize_pipeline(image: np.ndarray, 
                       name: str, 
                       output_dir: str = "./visualization"):
    """
    2D → 3D 변환 파이프라인 결과를 시각화합니다.
    
    Args:
        image: 입력 이미지
        name: 이미지 이름
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 방법으로 깊이 맵 생성
    methods = ["gradient", "intensity", "edge"]
    depth_maps = {}
    colored_maps = {}
    
    for method in methods:
        depth_maps[method] = generate_depth_map(image, method=method)
        colored_maps[method] = apply_colormap(depth_maps[method])
    
    # 시각화 Figure 생성
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'2D → 3D Conversion Pipeline: {name}', fontsize=14, fontweight='bold')
    
    # 원본 이미지
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    axes[2, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Original Image')
    axes[2, 0].axis('off')
    
    # 각 방법별 결과
    for idx, method in enumerate(methods):
        # 깊이 맵 (grayscale)
        axes[idx, 1].imshow(depth_maps[method], cmap='gray')
        axes[idx, 1].set_title(f'Depth Map ({method})')
        axes[idx, 1].axis('off')
        
        # 컬러 깊이 맵
        axes[idx, 2].imshow(cv2.cvtColor(colored_maps[method], cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title(f'Colored Depth ({method})')
        axes[idx, 2].axis('off')
        
        # 3D 포인트 미리보기 (2D 표현)
        points = convert_to_3d_points(depth_maps[method], downsample=4)
        scatter = axes[idx, 3].scatter(
            points[:, 0], points[:, 1], 
            c=points[:, 2], 
            cmap='jet', 
            s=1, 
            alpha=0.7
        )
        axes[idx, 3].set_title(f'3D Points Preview ({method})')
        axes[idx, 3].set_xlabel('X')
        axes[idx, 3].set_ylabel('Y')
        axes[idx, 3].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"저장됨: {output_dir}/{name}_pipeline.png")


def visualize_3d_surface(depth_map: np.ndarray, 
                         name: str, 
                         output_dir: str = "./visualization"):
    """
    깊이 맵을 3D 표면으로 시각화합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 다운샘플링 (시각화 성능)
    downsample = 4
    depth_small = depth_map[::downsample, ::downsample]
    h, w = depth_small.shape
    
    # 메쉬그리드 생성
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_small.astype(np.float32)
    
    # 3D Figure 생성
    fig = plt.figure(figsize=(14, 5))
    
    # 뷰 1: 정면
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('3D Surface (Front View)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Depth')
    ax1.view_init(elev=30, azim=45)
    
    # 뷰 2: 측면
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.set_title('3D Surface (Side View)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Depth')
    ax2.view_init(elev=15, azim=120)
    
    # 뷰 3: 위에서
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3.set_title('3D Surface (Top View)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Depth')
    ax3.view_init(elev=75, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_3d_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"저장됨: {output_dir}/{name}_3d_surface.png")


def create_comparison_figure(samples: dict, output_dir: str = "./visualization"):
    """
    여러 샘플의 비교 Figure 생성
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    fig.suptitle('2D → 3D Conversion Results Comparison', fontsize=16, fontweight='bold')
    
    for idx, (name, image) in enumerate(samples.items()):
        # 원본
        axes[idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'{name}\nOriginal')
        axes[idx, 0].axis('off')
        
        # 깊이 맵
        depth = generate_depth_map(image, method="gradient")
        axes[idx, 1].imshow(depth, cmap='gray')
        axes[idx, 1].set_title('Depth Map')
        axes[idx, 1].axis('off')
        
        # 컬러 깊이 맵
        colored = apply_colormap(depth)
        axes[idx, 2].imshow(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title('Colored Depth')
        axes[idx, 2].axis('off')
        
        # 3D 포인트
        points = convert_to_3d_points(depth, downsample=4)
        axes[idx, 3].scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                            cmap='jet', s=1, alpha=0.7)
        axes[idx, 3].set_title('3D Points')
        axes[idx, 3].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"저장됨: {output_dir}/comparison.png")


def main():
    """메인 실행 함수"""
    output_dir = "./visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("2D → 3D 변환 시각화 데모")
    print("=" * 60)
    
    # 샘플 이미지 생성
    print("\n1. 샘플 이미지 생성 중...")
    samples = create_sample_images()
    
    # 샘플 이미지 저장
    for name, image in samples.items():
        cv2.imwrite(os.path.join(output_dir, f'{name}_input.png'), image)
    print(f"   {len(samples)}개의 샘플 이미지 생성 완료")
    
    # 각 샘플에 대해 파이프라인 시각화
    print("\n2. 파이프라인 시각화 중...")
    for name, image in samples.items():
        visualize_pipeline(image, name, output_dir)
    
    # 3D 표면 시각화
    print("\n3. 3D 표면 시각화 중...")
    for name, image in samples.items():
        depth = generate_depth_map(image, method="gradient")
        visualize_3d_surface(depth, name, output_dir)
    
    # 비교 Figure 생성
    print("\n4. 비교 Figure 생성 중...")
    create_comparison_figure(samples, output_dir)
    
    print("\n" + "=" * 60)
    print("시각화 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    main()
