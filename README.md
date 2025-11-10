# Computer Vision - Project 1: Extended Otsu's Method

This project implements Otsu's method for automatic thresholding of grayscale images. Otsu's method is a classic computer vision technique that automatically determines an optimal threshold value to separate an image into foreground and background regions.

## Overview

In the original Otsu's method for automatic thresholding, we seek to find a threshold `t` that minimizes the weighted sum of within-group variances **σ²_B(t)** and **σ²_F(t)** for the background and foreground pixels that result from thresholding the grayscale image at value `t`.

This is equivalent to maximizing the between-class variance, which leads to the best separation between the two classes (background and foreground).

## Features

- **Automatic threshold detection** using Otsu's method
- **Within-class variance calculation** for background and foreground
- **Binary image generation** based on computed threshold
- **Visualization tools** for results analysis
- **Synthetic image generation** for testing and demonstration

## Mathematical Background

Otsu's method works by:

1. Computing the histogram of the grayscale image
2. Calculating the between-class variance for all possible threshold values
3. Selecting the threshold that maximizes the between-class variance

The between-class variance is defined as:
```
σ²_between(t) = ω₀(t) * ω₁(t) * (μ₀(t) - μ₁(t))²
```

Where:
- `ω₀(t)` and `ω₁(t)` are the probabilities of the background and foreground classes
- `μ₀(t)` and `μ₁(t)` are the mean intensities of the background and foreground classes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dongmo1-sta/Computer-Vision---Project-1--Extended-Otsu-s-method-.git
cd Computer-Vision---Project-1--Extended-Otsu-s-method-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import numpy as np
from otsu_thresholding import otsu_method

# Load or create a grayscale image
image = np.array(...)  # Your grayscale image

# Apply Otsu's method
threshold, binary_image, stats = otsu_method(image, return_stats=True)

print(f"Optimal threshold: {threshold}")
print(f"Background variance: {stats['background_variance']}")
print(f"Foreground variance: {stats['foreground_variance']}")
```

### Running the Demo

The project includes a demonstration script that creates a synthetic bimodal image and applies Otsu's method:

```bash
python demo.py
```

This will:
1. Create a synthetic grayscale image with two distinct regions
2. Apply Otsu's method to find the optimal threshold
3. Generate a visualization showing the original image, histogram with threshold, and binary result
4. Save the visualization as `otsu_result.png`

## Project Structure

```
.
├── README.md                  # This file
├── requirements.txt          # Python dependencies
├── otsu_thresholding.py     # Core implementation of Otsu's method
└── demo.py                  # Demonstration script
```

## Implementation Details

### Core Functions

- `otsu_threshold(image)`: Computes the optimal threshold value
- `calculate_within_class_variances(image, threshold)`: Calculates σ²_B and σ²_F
- `apply_threshold(image, threshold)`: Creates binary image
- `otsu_method(image, return_stats)`: Complete pipeline with optional statistics

### Algorithm Steps

1. Normalize the image histogram to obtain probability distribution
2. Compute cumulative sums for probabilities and means
3. For each possible threshold value t:
   - Calculate class probabilities ω₀(t) and ω₁(t)
   - Calculate class means μ₀(t) and μ₁(t)
   - Compute between-class variance σ²_between(t)
4. Select threshold that maximizes σ²_between(t)

## Requirements

- Python 3.6+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0 (for visualization)

## References

- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms". IEEE Transactions on Systems, Man, and Cybernetics. 9 (1): 62–66.

## License

This project is available for educational purposes.

## Author

Computer Vision Course - Project 1