"""
2D to 3D 변환 모듈의 Unit Test
pytest를 사용하여 코드의 기능을 검증합니다.

실행 방법:
    pytest test_depth_3d_converter.py -v

Author: Shin
Date: 2025-01-09
"""

import numpy as np
import pytest
import cv2
import os
import sys

# 모듈 import
from depth_3d_converter import (
    generate_depth_map,
    apply_colormap,
    convert_to_3d_points,
    save_point_cloud_ply,
    process_2d_to_3d
)


# ============================================================================
# Fixtures - 테스트에 사용할 공통 데이터
# ============================================================================

@pytest.fixture
def sample_color_image():
    """테스트용 컬러 이미지 (BGR 형식)"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 그라데이션 패턴
    for i in range(100):
        image[:, i] = [i * 2, 255 - i * 2, 128]
    return image


@pytest.fixture
def sample_grayscale_image():
    """테스트용 그레이스케일 이미지"""
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        image[:, i] = i * 2
    return image


@pytest.fixture
def sample_depth_map():
    """테스트용 깊이 맵"""
    depth = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            depth[i, j] = int(np.sqrt(i**2 + j**2) / np.sqrt(2) * 2.55)
    return depth


@pytest.fixture
def temp_output_dir(tmp_path):
    """임시 출력 디렉토리"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def temp_image_file(tmp_path, sample_color_image):
    """임시 이미지 파일"""
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), sample_color_image)
    return str(image_path)


# ============================================================================
# generate_depth_map() 테스트
# ============================================================================

class TestGenerateDepthMap:
    """generate_depth_map 함수 테스트 클래스"""
    
    def test_basic_functionality(self, sample_color_image):
        """기본 기능 테스트 - 깊이 맵 생성"""
        depth_map = generate_depth_map(sample_color_image)
        
        assert depth_map is not None, "깊이 맵이 None입니다."
        assert isinstance(depth_map, np.ndarray), "출력이 ndarray가 아닙니다."
        assert len(depth_map.shape) == 2, "깊이 맵은 2D 배열이어야 합니다."
    
    def test_output_shape(self, sample_color_image):
        """출력 크기가 입력과 동일한지 검증"""
        depth_map = generate_depth_map(sample_color_image)
        
        expected_shape = (sample_color_image.shape[0], sample_color_image.shape[1])
        assert depth_map.shape == expected_shape, \
            f"출력 크기 {depth_map.shape}가 예상 {expected_shape}와 다릅니다."
    
    def test_output_dtype(self, sample_color_image):
        """출력 데이터 타입 검증"""
        depth_map = generate_depth_map(sample_color_image)
        
        assert depth_map.dtype == np.uint8, \
            f"출력 dtype {depth_map.dtype}이 uint8이 아닙니다."
    
    def test_grayscale_input(self, sample_grayscale_image):
        """그레이스케일 입력 처리 테스트"""
        depth_map = generate_depth_map(sample_grayscale_image)
        
        assert depth_map is not None
        assert depth_map.shape == sample_grayscale_image.shape
    
    def test_none_input_raises_error(self):
        """None 입력시 ValueError 발생 확인"""
        with pytest.raises(ValueError, match="입력된 이미지가 없습니다"):
            generate_depth_map(None)
    
    def test_invalid_type_raises_error(self):
        """잘못된 타입 입력시 TypeError 발생 확인"""
        with pytest.raises(TypeError, match="numpy ndarray"):
            generate_depth_map("not_an_image")
    
    def test_empty_image_raises_error(self):
        """빈 이미지 입력시 ValueError 발생 확인"""
        empty_image = np.array([])
        with pytest.raises(ValueError, match="빈 이미지"):
            generate_depth_map(empty_image)
    
    def test_gradient_method(self, sample_color_image):
        """gradient 방법 테스트"""
        depth_map = generate_depth_map(sample_color_image, method="gradient")
        assert depth_map is not None
    
    def test_intensity_method(self, sample_color_image):
        """intensity 방법 테스트"""
        depth_map = generate_depth_map(sample_color_image, method="intensity")
        assert depth_map is not None
    
    def test_edge_method(self, sample_color_image):
        """edge 방법 테스트"""
        depth_map = generate_depth_map(sample_color_image, method="edge")
        assert depth_map is not None
    
    def test_invalid_method_raises_error(self, sample_color_image):
        """지원하지 않는 방법 입력시 에러 발생"""
        with pytest.raises(ValueError, match="지원하지 않는 방법"):
            generate_depth_map(sample_color_image, method="invalid_method")
    
    def test_different_image_sizes(self):
        """다양한 이미지 크기 테스트"""
        sizes = [(50, 50), (100, 200), (200, 100), (256, 256)]
        
        for h, w in sizes:
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            depth_map = generate_depth_map(image)
            assert depth_map.shape == (h, w), \
                f"크기 {(h, w)}에서 출력 크기가 일치하지 않습니다."


# ============================================================================
# apply_colormap() 테스트
# ============================================================================

class TestApplyColormap:
    """apply_colormap 함수 테스트 클래스"""
    
    def test_basic_functionality(self, sample_depth_map):
        """기본 기능 테스트"""
        colored = apply_colormap(sample_depth_map)
        
        assert colored is not None
        assert isinstance(colored, np.ndarray)
    
    def test_output_is_color(self, sample_depth_map):
        """출력이 컬러 이미지(3채널)인지 확인"""
        colored = apply_colormap(sample_depth_map)
        
        assert len(colored.shape) == 3, "출력이 3D 배열이 아닙니다."
        assert colored.shape[2] == 3, "출력이 3채널이 아닙니다."
    
    def test_output_shape_matches_input(self, sample_depth_map):
        """출력 크기가 입력과 일치하는지 확인"""
        colored = apply_colormap(sample_depth_map)
        
        expected_shape = (sample_depth_map.shape[0], sample_depth_map.shape[1], 3)
        assert colored.shape == expected_shape
    
    def test_none_input_raises_error(self):
        """None 입력시 에러 발생"""
        with pytest.raises(ValueError):
            apply_colormap(None)
    
    def test_invalid_type_raises_error(self):
        """잘못된 타입 입력시 에러 발생"""
        with pytest.raises(TypeError):
            apply_colormap([1, 2, 3])
    
    def test_3d_input_raises_error(self):
        """3D 배열 입력시 에러 발생"""
        invalid_input = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D 배열"):
            apply_colormap(invalid_input)
    
    def test_different_colormaps(self, sample_depth_map):
        """다양한 컬러맵 테스트"""
        colormaps = [
            cv2.COLORMAP_JET,
            cv2.COLORMAP_HOT,
            cv2.COLORMAP_VIRIDIS,
            cv2.COLORMAP_RAINBOW
        ]
        
        for cmap in colormaps:
            colored = apply_colormap(sample_depth_map, colormap=cmap)
            assert colored is not None
            assert colored.shape[2] == 3


# ============================================================================
# convert_to_3d_points() 테스트
# ============================================================================

class TestConvertTo3DPoints:
    """convert_to_3d_points 함수 테스트 클래스"""
    
    def test_basic_functionality(self, sample_depth_map):
        """기본 기능 테스트"""
        points = convert_to_3d_points(sample_depth_map)
        
        assert points is not None
        assert isinstance(points, np.ndarray)
    
    def test_output_shape(self, sample_depth_map):
        """출력 형태 검증 - (N, 3)"""
        points = convert_to_3d_points(sample_depth_map)
        
        h, w = sample_depth_map.shape
        expected_num_points = h * w
        
        assert points.shape == (expected_num_points, 3), \
            f"출력 shape {points.shape}가 예상 ({expected_num_points}, 3)와 다릅니다."
    
    def test_output_contains_xyz(self, sample_depth_map):
        """출력이 X, Y, Z 좌표를 포함하는지 확인"""
        points = convert_to_3d_points(sample_depth_map)
        
        # 각 포인트는 3개의 값을 가져야 함
        assert points.shape[1] == 3
        
        # X, Y 좌표 범위 확인
        h, w = sample_depth_map.shape
        assert points[:, 0].min() >= 0  # X min
        assert points[:, 0].max() < w   # X max
        assert points[:, 1].min() >= 0  # Y min
        assert points[:, 1].max() < h   # Y max
    
    def test_none_input_raises_error(self):
        """None 입력시 에러 발생"""
        with pytest.raises(ValueError):
            convert_to_3d_points(None)
    
    def test_invalid_type_raises_error(self):
        """잘못된 타입 입력시 에러 발생"""
        with pytest.raises(TypeError):
            convert_to_3d_points("invalid")
    
    def test_scale_z_parameter(self, sample_depth_map):
        """scale_z 파라미터 테스트"""
        points_scale1 = convert_to_3d_points(sample_depth_map, scale_z=1.0)
        points_scale2 = convert_to_3d_points(sample_depth_map, scale_z=2.0)
        
        # Z값이 스케일에 따라 변경되는지 확인
        z_ratio = points_scale2[:, 2].mean() / (points_scale1[:, 2].mean() + 1e-10)
        assert abs(z_ratio - 2.0) < 0.1, "scale_z가 올바르게 적용되지 않았습니다."
    
    def test_invalid_scale_z_raises_error(self, sample_depth_map):
        """잘못된 scale_z 값 테스트"""
        with pytest.raises(ValueError, match="양수"):
            convert_to_3d_points(sample_depth_map, scale_z=-1.0)
        
        with pytest.raises(ValueError, match="양수"):
            convert_to_3d_points(sample_depth_map, scale_z=0)
    
    def test_downsample_parameter(self, sample_depth_map):
        """downsample 파라미터 테스트"""
        points_full = convert_to_3d_points(sample_depth_map, downsample=1)
        points_half = convert_to_3d_points(sample_depth_map, downsample=2)
        
        # 다운샘플링시 포인트 수가 감소해야 함
        assert len(points_half) < len(points_full)
        
        # 대략 1/4이 되어야 함 (2x2 다운샘플링)
        ratio = len(points_half) / len(points_full)
        assert 0.2 < ratio < 0.3, "다운샘플링 비율이 예상과 다릅니다."
    
    def test_invalid_downsample_raises_error(self, sample_depth_map):
        """잘못된 downsample 값 테스트"""
        with pytest.raises(ValueError, match="1 이상"):
            convert_to_3d_points(sample_depth_map, downsample=0)


# ============================================================================
# save_point_cloud_ply() 테스트
# ============================================================================

class TestSavePointCloudPLY:
    """save_point_cloud_ply 함수 테스트 클래스"""
    
    def test_basic_save(self, temp_output_dir):
        """기본 저장 기능 테스트"""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        filename = os.path.join(temp_output_dir, "test.ply")
        
        result = save_point_cloud_ply(points, None, filename)
        
        assert result == True
        assert os.path.exists(filename)
    
    def test_save_with_colors(self, temp_output_dir):
        """색상 포함 저장 테스트"""
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        filename = os.path.join(temp_output_dir, "test_colors.ply")
        
        result = save_point_cloud_ply(points, colors, filename)
        
        assert result == True
        assert os.path.exists(filename)
        
        # 파일 내용 확인
        with open(filename, 'r') as f:
            content = f.read()
            assert "property uchar red" in content
            assert "property uchar green" in content
            assert "property uchar blue" in content
    
    def test_ply_header_format(self, temp_output_dir):
        """PLY 헤더 형식 검증"""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        filename = os.path.join(temp_output_dir, "test_header.ply")
        
        save_point_cloud_ply(points, None, filename)
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == "ply"
            assert "format ascii 1.0" in lines[1]
            assert "element vertex 1" in lines[2]
    
    def test_empty_points_raises_error(self, temp_output_dir):
        """빈 포인트 배열 테스트"""
        points = np.array([])
        filename = os.path.join(temp_output_dir, "empty.ply")
        
        with pytest.raises(ValueError, match="비어있습니다"):
            save_point_cloud_ply(points, None, filename)
    
    def test_none_points_raises_error(self, temp_output_dir):
        """None 입력 테스트"""
        filename = os.path.join(temp_output_dir, "none.ply")
        
        with pytest.raises(ValueError):
            save_point_cloud_ply(None, None, filename)


# ============================================================================
# process_2d_to_3d() 통합 테스트
# ============================================================================

class TestProcess2DTo3D:
    """process_2d_to_3d 파이프라인 통합 테스트"""
    
    def test_full_pipeline(self, temp_image_file, temp_output_dir):
        """전체 파이프라인 테스트"""
        result = process_2d_to_3d(temp_image_file, temp_output_dir)
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_output_files_created(self, temp_image_file, temp_output_dir):
        """출력 파일 생성 확인"""
        result = process_2d_to_3d(temp_image_file, temp_output_dir)
        
        assert os.path.exists(result['depth_path'])
        assert os.path.exists(result['colored_path'])
        assert os.path.exists(result['ply_path'])
    
    def test_result_keys(self, temp_image_file, temp_output_dir):
        """결과 딕셔너리 키 확인"""
        result = process_2d_to_3d(temp_image_file, temp_output_dir)
        
        expected_keys = [
            'original_shape', 
            'depth_map_shape', 
            'num_3d_points',
            'depth_path',
            'colored_path',
            'ply_path'
        ]
        
        for key in expected_keys:
            assert key in result, f"결과에 '{key}' 키가 없습니다."
    
    def test_invalid_image_path_raises_error(self, temp_output_dir):
        """존재하지 않는 이미지 경로 테스트"""
        with pytest.raises(FileNotFoundError):
            process_2d_to_3d("nonexistent_image.jpg", temp_output_dir)
    
    def test_different_depth_methods(self, temp_image_file, temp_output_dir):
        """다양한 깊이 추정 방법 테스트"""
        methods = ["gradient", "intensity", "edge"]
        
        for method in methods:
            output_dir = os.path.join(temp_output_dir, method)
            result = process_2d_to_3d(
                temp_image_file, 
                output_dir, 
                depth_method=method
            )
            assert result is not None, f"{method} 방법 실패"


# ============================================================================
# 성능 및 엣지 케이스 테스트
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 및 경계 조건 테스트"""
    
    def test_single_pixel_image(self):
        """1x1 이미지 처리"""
        image = np.array([[[128, 128, 128]]], dtype=np.uint8)
        depth_map = generate_depth_map(image)
        
        assert depth_map.shape == (1, 1)
    
    def test_very_large_image(self):
        """큰 이미지 처리 (메모리 테스트)"""
        # 500x500 이미지 테스트
        image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        
        depth_map = generate_depth_map(image)
        assert depth_map.shape == (500, 500)
        
        points = convert_to_3d_points(depth_map, downsample=4)
        assert len(points) > 0
    
    def test_uniform_image(self):
        """균일한 이미지 (모든 픽셀 동일)"""
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        depth_map = generate_depth_map(image)
        
        assert depth_map is not None
    
    def test_black_image(self):
        """검정 이미지"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        depth_map = generate_depth_map(image)
        
        assert depth_map is not None
    
    def test_white_image(self):
        """흰색 이미지"""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        depth_map = generate_depth_map(image)
        
        assert depth_map is not None


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v", "--tb=short"])
