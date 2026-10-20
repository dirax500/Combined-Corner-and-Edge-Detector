# Harris Corner Detector

The project implements the Harris & Stephens (1988) corner detection algorithm by using pure NumPy — no OpenCV, no SciPy.

**Reference:** Harris, C. & Stephens, M. (1988). *A Combined Corner and Edge Detector.* Proceedings of the 4th Alvey Vision Conference.

---

## Overview

Corner detection is one of the core tasks in computer vision, used in image matching, object tracking, and 3D reconstruction. This implementation follows the original Harris & Stephens paper:

1. Compute image gradients with Sobel operators
2. Build the second-moment (structure) tensor at each pixel
3. Smooth the tensor with a Gaussian window
4. Compute the corner response function `R = det(M) − k · trace(M)²`
5. Threshold and apply non-maximum suppression to find final corners

The response `R` classifies each pixel:
- `R >> 0` → corner
- `R << 0` → edge
- `|R| ≈ 0` → flat region

---

## Requirements

- Python 3.7+
- numpy
- matplotlib
- Pillow

Install dependencies:

```bash
pip install numpy matplotlib Pillow
```

---

## Usage

**Run the built-in demo** (synthetic checkerboard + shapes + a downloaded sample image):

```bash
python harris.py
```

**Run on your own image:**

```bash
python harris.py path/to/your/image.jpg
```

Output PNGs are saved to the same directory as the script.

---

## API

### `detect_corners(gray, k, threshold_ratio, window_size, sigma, nms_size)`

The main entry point. Takes a float64 grayscale array and returns `(corner_mask, R)`.

| Parameter | Default | Description |
|---|---|---|
| `k` | `0.04` | Harris sensitivity constant (0.04–0.06 typical) |
| `threshold_ratio` | `0.01` | Fraction of `R.max()` used as the detection threshold |
| `window_size` | `5` | Gaussian smoothing kernel size |
| `sigma` | `1.5` | Gaussian standard deviation |
| `nms_size` | `7` | Non-maximum suppression neighbourhood size |

### `visualise_pipeline(img_array, name, save_path, **kwargs)`

Produces a 6-panel figure showing every stage of the pipeline: original image, grayscale, gradient magnitude, corner response map, detected corner mask, and corners overlaid on the original.

### Lower-level functions

| Function | Description |
|---|---|
| `rgb_to_grayscale(img)` | Converts RGB to grayscale using ITU-R BT.709 luminance weights |
| `convolve2d(image, kernel)` | Manual 2D convolution via NumPy stride tricks |
| `gaussian_kernel(size, sigma)` | Generates a normalised Gaussian kernel |
| `sobel_gradients(gray)` | Computes Ix, Iy using Sobel operators |
| `harris_response(gray, k, ...)` | Computes the full Harris R map |
| `non_max_suppression(R, neighborhood)` | Retains only local maxima |
| `draw_corners(img, corners, ...)` | Overlays detected corners as red circles |

---

## Demo Outputs

The demo runs detection on three test cases and saves a PNG for each:

| File | Description |
|---|---|
| `output_checkerboard.png` | 8×8 checkerboard — high corner density |
| `output_shapes.png` | Rectangle, triangle, circle — mixed feature types |
| `output_real.png` | Real-world photograph |

---

## Implementation Notes

- **No external vision libraries.** Convolution is implemented from scratch using `numpy.lib.stride_tricks.as_strided` for efficiency.
- **Reflect padding** is used in convolution to avoid border artifacts.
- **Non-maximum suppression** is implemented as a sliding-window max, equivalent to a morphological dilation followed by an equality check.
- Tuning `threshold_ratio` lower (e.g. `0.005`) detects more corners in complex real images; higher values keep only the strongest responses.
