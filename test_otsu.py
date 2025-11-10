"""
Unit tests for Otsu's thresholding method.

These tests verify the correctness of the implementation.
"""

import numpy as np
from otsu_thresholding import (
    otsu_threshold, 
    calculate_within_class_variances,
    apply_threshold,
    otsu_method
)


def test_uniform_image():
    """Test with uniform image - should return threshold near middle."""
    image = np.ones((100, 100), dtype=np.uint8) * 128
    threshold = otsu_threshold(image)
    # For uniform image, any threshold is valid since there's no variation
    assert 0 <= threshold <= 255, f"Threshold {threshold} out of range"
    print("✓ Uniform image test passed")


def test_binary_image():
    """Test with already binary image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50:, :] = 255  # Bottom half is white
    
    threshold = otsu_threshold(image)
    # Threshold should be between 0 and 255
    assert 0 <= threshold < 255, f"Threshold {threshold} should be between 0 and 255"
    print(f"✓ Binary image test passed (threshold={threshold})")


def test_bimodal_image():
    """Test with bimodal distribution."""
    np.random.seed(42)
    
    # Create bimodal image
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Background: mean=50, foreground: mean=200
    background = np.random.normal(50, 10, (200, 100)).astype(np.uint8)
    foreground = np.random.normal(200, 10, (200, 100)).astype(np.uint8)
    
    image[:, :100] = np.clip(background, 0, 255)
    image[:, 100:] = np.clip(foreground, 0, 255)
    
    threshold = otsu_threshold(image)
    
    # Threshold should be somewhere between the two peaks (roughly 50-200)
    assert 75 <= threshold <= 175, f"Threshold {threshold} not between modes"
    print(f"✓ Bimodal image test passed (threshold={threshold})")


def test_within_class_variances():
    """Test calculation of within-class variances."""
    # Simple test image
    image = np.array([[0, 0, 100, 100],
                      [0, 0, 100, 100]], dtype=np.uint8)
    
    threshold = 50
    bg_var, fg_var = calculate_within_class_variances(image, threshold)
    
    # Background pixels (0) should have variance 0
    # Foreground pixels (100) should have variance 0
    assert bg_var == 0.0, f"Background variance should be 0, got {bg_var}"
    assert fg_var == 0.0, f"Foreground variance should be 0, got {fg_var}"
    print("✓ Within-class variances test passed")


def test_apply_threshold():
    """Test threshold application."""
    image = np.array([[0, 50, 100, 150, 200]], dtype=np.uint8)
    threshold = 100
    
    binary = apply_threshold(image, threshold)
    
    # Pixels <= 100 should be 0, pixels > 100 should be 255
    expected = np.array([[0, 0, 0, 255, 255]], dtype=np.uint8)
    assert np.array_equal(binary, expected), "Binary image incorrect"
    print("✓ Apply threshold test passed")


def test_complete_pipeline():
    """Test the complete otsu_method function."""
    np.random.seed(42)
    
    # Create test image
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    threshold, binary_image, stats = otsu_method(image, return_stats=True)
    
    # Verify outputs
    assert isinstance(threshold, (int, np.integer)), "Threshold should be integer"
    assert binary_image.shape == image.shape, "Binary image shape mismatch"
    assert stats is not None, "Stats should be returned"
    assert 'threshold' in stats, "Stats should contain threshold"
    assert 'background_variance' in stats, "Stats should contain background_variance"
    assert 'foreground_variance' in stats, "Stats should contain foreground_variance"
    
    # Binary image should only have 0 and 255
    unique_values = np.unique(binary_image)
    assert all(v in [0, 255] for v in unique_values), "Binary image should only have 0 and 255"
    
    print(f"✓ Complete pipeline test passed (threshold={threshold})")


def test_normalized_image():
    """Test with normalized image (0-1 range)."""
    image = np.random.rand(50, 50)  # Values between 0 and 1
    
    threshold = otsu_threshold(image)
    
    # Should still work and return threshold in 0-255 range
    assert 0 <= threshold <= 255, f"Threshold {threshold} out of range"
    print(f"✓ Normalized image test passed (threshold={threshold})")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Otsu's Method Tests")
    print("=" * 70)
    print()
    
    test_uniform_image()
    test_binary_image()
    test_bimodal_image()
    test_within_class_variances()
    test_apply_threshold()
    test_complete_pipeline()
    test_normalized_image()
    
    print()
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
