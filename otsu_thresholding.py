"""
Otsu's Method for Automatic Thresholding

This module implements Otsu's method for automatic thresholding of grayscale images.
The method finds an optimal threshold t that minimizes the weighted sum of within-group 
variances for the background (σ²_B(t)) and foreground (σ²_F(t)) pixels.
"""

import numpy as np
from typing import Tuple, Optional


def calculate_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the histogram of a grayscale image.
    
    Args:
        image: Grayscale image as numpy array
        bins: Number of bins for histogram (default 256 for 8-bit images)
    
    Returns:
        Tuple of (histogram, bin_edges)
    """
    histogram, bin_edges = np.histogram(image.flatten(), bins=bins, range=(0, bins))
    return histogram, bin_edges


def otsu_threshold(image: np.ndarray) -> int:
    """
    Compute the optimal threshold using Otsu's method.
    
    Otsu's method finds the threshold t that minimizes the weighted sum of 
    within-class variances for background and foreground pixels.
    
    This is equivalent to maximizing the between-class variance.
    
    Args:
        image: Grayscale image as numpy array
    
    Returns:
        Optimal threshold value
    """
    # Normalize image to 0-255 range if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Calculate histogram
    histogram, _ = calculate_histogram(image)
    
    # Normalize histogram to get probabilities
    histogram = histogram.astype(float)
    total_pixels = histogram.sum()
    prob = histogram / total_pixels
    
    # Calculate cumulative sums
    omega = np.cumsum(prob)  # Cumulative probability
    mu = np.cumsum(prob * np.arange(256))  # Cumulative mean
    mu_total = mu[-1]
    
    # Calculate between-class variance for all possible thresholds
    # Between-class variance = omega_0 * omega_1 * (mu_0 - mu_1)^2
    # where omega_0 and omega_1 are the cumulative probabilities
    # and mu_0 and mu_1 are the class means
    
    sigma_between = np.zeros(256)
    
    for t in range(256):
        omega_0 = omega[t]  # Background cumulative probability
        omega_1 = 1.0 - omega_0  # Foreground cumulative probability
        
        if omega_0 == 0 or omega_1 == 0:
            continue
        
        mu_0 = mu[t] / omega_0  # Background mean
        mu_1 = (mu_total - mu[t]) / omega_1  # Foreground mean
        
        # Between-class variance
        sigma_between[t] = omega_0 * omega_1 * (mu_0 - mu_1) ** 2
    
    # Find threshold that maximizes between-class variance
    optimal_threshold = np.argmax(sigma_between)
    
    return optimal_threshold


def calculate_within_class_variances(image: np.ndarray, threshold: int) -> Tuple[float, float]:
    """
    Calculate the within-class variances for background and foreground.
    
    Args:
        image: Grayscale image as numpy array
        threshold: Threshold value to separate background and foreground
    
    Returns:
        Tuple of (background_variance, foreground_variance)
    """
    # Normalize image to 0-255 range if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Separate pixels into background and foreground
    background_pixels = image[image <= threshold]
    foreground_pixels = image[image > threshold]
    
    # Calculate variances
    if len(background_pixels) > 0:
        background_variance = np.var(background_pixels)
    else:
        background_variance = 0.0
    
    if len(foreground_pixels) > 0:
        foreground_variance = np.var(foreground_pixels)
    else:
        foreground_variance = 0.0
    
    return background_variance, foreground_variance


def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Apply threshold to create binary image.
    
    Args:
        image: Grayscale image as numpy array
        threshold: Threshold value
    
    Returns:
        Binary image (0 for background, 255 for foreground)
    """
    # Normalize image to 0-255 range if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    
    return binary_image


def otsu_method(image: np.ndarray, return_stats: bool = False) -> Tuple[int, np.ndarray, Optional[dict]]:
    """
    Complete Otsu's method implementation.
    
    Args:
        image: Grayscale image as numpy array
        return_stats: If True, return statistics about the thresholding
    
    Returns:
        Tuple of (threshold, binary_image, stats_dict)
        stats_dict contains background and foreground variances if return_stats=True
    """
    # Find optimal threshold
    threshold = otsu_threshold(image)
    
    # Apply threshold
    binary_image = apply_threshold(image, threshold)
    
    # Calculate statistics if requested
    stats = None
    if return_stats:
        bg_var, fg_var = calculate_within_class_variances(image, threshold)
        stats = {
            'threshold': threshold,
            'background_variance': bg_var,
            'foreground_variance': fg_var,
            'weighted_within_class_variance': bg_var + fg_var
        }
    
    return threshold, binary_image, stats


if __name__ == "__main__":
    # Example usage
    print("Otsu's Method for Automatic Thresholding")
    print("=" * 50)
    print("\nThis module implements Otsu's method for finding optimal")
    print("threshold values in grayscale images.")
    print("\nUsage:")
    print("  from otsu_thresholding import otsu_method")
    print("  threshold, binary_image, stats = otsu_method(image, return_stats=True)")
