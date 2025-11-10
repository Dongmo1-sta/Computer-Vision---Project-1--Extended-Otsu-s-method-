# Otsu's Method - Quick Reference

## What is Otsu's Method?

Otsu's method is an automatic image thresholding technique that separates a grayscale image into foreground and background by finding an optimal threshold value.

## Key Concepts

### Objective
Find threshold `t` that **minimizes** the weighted sum of within-class variances:
- **σ²_B(t)**: Variance of background pixels (pixels ≤ t)
- **σ²_F(t)**: Variance of foreground pixels (pixels > t)

This is equivalent to **maximizing** the between-class variance.

### Mathematical Formulation

The between-class variance is:
```
σ²_between(t) = ω₀(t) × ω₁(t) × [μ₀(t) - μ₁(t)]²
```

Where:
- `ω₀(t)` = probability of background class (weight)
- `ω₁(t)` = probability of foreground class (weight)
- `μ₀(t)` = mean intensity of background
- `μ₁(t)` = mean intensity of foreground

## Quick Start

```python
from otsu_thresholding import otsu_method
import numpy as np

# Your grayscale image (0-255)
image = np.array(...)

# Apply Otsu's method
threshold, binary_image, stats = otsu_method(image, return_stats=True)

print(f"Optimal threshold: {threshold}")
print(f"Background variance: {stats['background_variance']:.2f}")
print(f"Foreground variance: {stats['foreground_variance']:.2f}")
```

## API Reference

### `otsu_threshold(image)`
**Purpose**: Find optimal threshold value  
**Input**: Grayscale image (numpy array)  
**Output**: Integer threshold value (0-255)

### `calculate_within_class_variances(image, threshold)`
**Purpose**: Calculate σ²_B and σ²_F  
**Input**: Image and threshold value  
**Output**: Tuple (background_variance, foreground_variance)

### `apply_threshold(image, threshold)`
**Purpose**: Create binary image  
**Input**: Image and threshold  
**Output**: Binary image (0=background, 255=foreground)

### `otsu_method(image, return_stats=False)`
**Purpose**: Complete pipeline  
**Input**: Image and optional stats flag  
**Output**: (threshold, binary_image, stats_dict)

## Algorithm Steps

1. **Compute histogram** of grayscale image
2. **Normalize** to probability distribution
3. **For each threshold t** (0-255):
   - Calculate class probabilities ω₀(t), ω₁(t)
   - Calculate class means μ₀(t), μ₁(t)
   - Compute between-class variance σ²_between(t)
4. **Select** threshold that maximizes σ²_between(t)

## When to Use Otsu's Method

### ✅ Good for:
- Images with **bimodal histograms** (two peaks)
- Automatic thresholding without user input
- Document scanning, QR codes, text extraction
- Object detection with clear background/foreground separation

### ❌ Not ideal for:
- Images with **unimodal histograms** (single peak)
- Multiple objects with varying intensities
- Images with uneven illumination
- Complex scenes with many objects

## Example Output

```
Optimal threshold: 119
Background variance (σ²_B): 228.80
Foreground variance (σ²_F): 396.67
Weighted within-class variance: 625.47
```

## Running Examples

```bash
# Run demo with synthetic image
python demo.py

# Run tests
python test_otsu.py

# Import in your code
python -c "from otsu_thresholding import otsu_method; help(otsu_method)"
```

## References

- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms". 
  IEEE Transactions on Systems, Man, and Cybernetics. 9(1): 62–66.
