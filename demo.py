"""
Demo script for Otsu's Method

This script demonstrates the usage of Otsu's thresholding method with synthetic images.
"""

import numpy as np
import matplotlib.pyplot as plt
from otsu_thresholding import otsu_method, otsu_threshold, calculate_within_class_variances


def create_synthetic_bimodal_image(size: tuple = (256, 256), seed: int = 42) -> np.ndarray:
    """
    Create a synthetic bimodal grayscale image for testing.
    
    Args:
        size: Image dimensions (height, width)
        seed: Random seed for reproducibility
    
    Returns:
        Synthetic grayscale image
    """
    np.random.seed(seed)
    
    # Create two regions with different intensities
    image = np.zeros(size)
    
    # Background region (darker)
    background_mean = 60
    background_std = 15
    background = np.random.normal(background_mean, background_std, size)
    
    # Foreground region (brighter)
    foreground_mean = 180
    foreground_std = 20
    foreground = np.random.normal(foreground_mean, foreground_std, size)
    
    # Create a simple geometric shape (circle) for foreground
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 3
    
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                image[i, j] = foreground[i, j]
            else:
                image[i, j] = background[i, j]
    
    # Clip values to valid range
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def visualize_otsu_result(image: np.ndarray, threshold: int, binary_image: np.ndarray, 
                          stats: dict = None, save_path: str = None):
    """
    Visualize the original image, threshold, and binary result.
    
    Args:
        image: Original grayscale image
        threshold: Computed threshold value
        binary_image: Binary thresholded image
        stats: Optional statistics dictionary
        save_path: Path to save the figure (if None, display only)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Histogram with threshold line
    axes[1].hist(image.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram with Optimal Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Binary image
    axes[2].imshow(binary_image, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Thresholded Binary Image')
    axes[2].axis('off')
    
    # Add statistics if available
    if stats:
        info_text = f"Threshold: {stats['threshold']}\n"
        info_text += f"Background Variance: {stats['background_variance']:.2f}\n"
        info_text += f"Foreground Variance: {stats['foreground_variance']:.2f}\n"
        info_text += f"Total Within-Class Variance: {stats['weighted_within_class_variance']:.2f}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_otsu_method():
    """
    Demonstrate Otsu's method on a synthetic image.
    """
    print("=" * 70)
    print("Otsu's Method for Automatic Thresholding - Demo")
    print("=" * 70)
    
    # Create synthetic bimodal image
    print("\n1. Creating synthetic bimodal image...")
    image = create_synthetic_bimodal_image(size=(256, 256))
    print(f"   Image shape: {image.shape}")
    print(f"   Image range: [{image.min()}, {image.max()}]")
    
    # Apply Otsu's method
    print("\n2. Applying Otsu's method...")
    threshold, binary_image, stats = otsu_method(image, return_stats=True)
    
    print(f"\n3. Results:")
    print(f"   Optimal threshold: {threshold}")
    print(f"   Background variance (σ²_B): {stats['background_variance']:.2f}")
    print(f"   Foreground variance (σ²_F): {stats['foreground_variance']:.2f}")
    print(f"   Weighted within-class variance: {stats['weighted_within_class_variance']:.2f}")
    
    # Calculate percentage of background and foreground pixels
    bg_pixels = np.sum(image <= threshold)
    fg_pixels = np.sum(image > threshold)
    total_pixels = image.size
    
    print(f"\n4. Pixel Statistics:")
    print(f"   Background pixels: {bg_pixels} ({100*bg_pixels/total_pixels:.1f}%)")
    print(f"   Foreground pixels: {fg_pixels} ({100*fg_pixels/total_pixels:.1f}%)")
    
    # Visualize results
    print("\n5. Generating visualization...")
    visualize_otsu_result(image, threshold, binary_image, stats, 
                         save_path='otsu_result.png')
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    demo_otsu_method()
