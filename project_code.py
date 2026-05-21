"""
Harris Corner Detector
Paper: "A Combined Corner and Edge Detector" by Chris Harris & Mike Stephens (1988)

Team: Beibarys, Alim, Didar, Dias

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────
#  1.  CORE PRIMITIVES
# ─────────────────────────────────────────────

def rgb_to_grayscale(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using standard luminance weights."""
    if img_array.ndim == 2:
        return img_array.astype(np.float64)
    return (0.2126 * img_array[:, :, 0] +
            0.7152 * img_array[:, :, 1] +
            0.0722 * img_array[:, :, 2]).astype(np.float64)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2-D convolution (same-size output, zero-padding).
    No scipy / cv2 – pure NumPy stride tricks for speed.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    # Use as_strided for an efficient sliding-window view
    from numpy.lib.stride_tricks import as_strided
    H, W = image.shape
    shape = (H, W, kh, kw)
    strides = (*padded.strides, *padded.strides)
    windows = as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Create a normalised Gaussian kernel."""
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def sobel_gradients(gray: np.ndarray):
    """
    Compute image gradients Ix, Iy using Sobel operators.
    Sobel kernels approximate the first derivative with noise suppression.
    """
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    Ix = convolve2d(gray, Kx)
    Iy = convolve2d(gray, Ky)
    return Ix, Iy


# ─────────────────────────────────────────────
#  2.  HARRIS CORNER RESPONSE
# ─────────────────────────────────────────────

def harris_response(gray: np.ndarray,
                    k: float = 0.04,
                    window_size: int = 5,
                    sigma: float = 1.0) -> np.ndarray:
    """
    Compute the Harris corner response R for every pixel.

    Algorithm (Harris & Stephens, 1988):
      1.  Compute gradients  Ix, Iy
      2.  Compute second-moment (structure) tensor components:
              Ixx = Ix²,  Iyy = Iy²,  Ixy = Ix·Iy
      3.  Smooth each component with a Gaussian window
      4.  R = det(M) − k · trace(M)²
              where  M = [[Sxx, Sxy],
                          [Sxy, Syy]]
      Interpretation:
          R >> 0  →  corner
          R << 0  →  edge
          |R| ≈ 0 →  flat region
    """
    Ix, Iy = sobel_gradients(gray)

    # Structure tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Weighted spatial sum via Gaussian smoothing
    gauss = gaussian_kernel(window_size, sigma)
    Sxx = convolve2d(Ixx, gauss)
    Syy = convolve2d(Iyy, gauss)
    Sxy = convolve2d(Ixy, gauss)

    # Corner response function
    det_M   = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    return R


# ─────────────────────────────────────────────
#  3.  NON-MAXIMUM SUPPRESSION + THRESHOLDING
# ─────────────────────────────────────────────

def non_max_suppression(R: np.ndarray, neighborhood: int = 5) -> np.ndarray:
    """
    Keep only local maxima within a (neighborhood × neighborhood) window.
    Equivalent to 'dilation + equality check'.
    """
    from numpy.lib.stride_tricks import as_strided

    half = neighborhood // 2
    padded = np.pad(R, half, mode='constant', constant_values=-np.inf)
    H, W = R.shape
    shape   = (H, W, neighborhood, neighborhood)
    strides = (*padded.strides, *padded.strides)
    windows = as_strided(padded, shape=shape, strides=strides)
    local_max = windows.max(axis=(2, 3))
    return (R == local_max) & (R > 0)


def detect_corners(gray: np.ndarray,
                   k: float = 0.04,
                   threshold_ratio: float = 0.01,
                   window_size: int = 5,
                   sigma: float = 1.5,
                   nms_size: int = 7):
    """
    Full Harris pipeline.  Returns (corner_mask, R).

    Parameters
    ----------
    gray            : float64 grayscale image
    k               : Harris sensitivity (0.04–0.06 typical)
    threshold_ratio : fraction of R.max() used as threshold
    window_size     : Gaussian smoothing kernel size
    sigma           : Gaussian std-dev
    nms_size        : non-max suppression neighbourhood
    """
    R = harris_response(gray, k=k, window_size=window_size, sigma=sigma)
    threshold = threshold_ratio * R.max()
    nms_mask  = non_max_suppression(R, neighborhood=nms_size)
    corners   = nms_mask & (R > threshold)
    return corners, R


# ─────────────────────────────────────────────
#  4.  VISUALISATION HELPERS
# ─────────────────────────────────────────────

def draw_corners(img_array: np.ndarray, corners: np.ndarray,
                 color=(255, 0, 0), radius: int = 4) -> np.ndarray:
    """Overlay red circles on corner locations (returns RGB copy)."""
    out = img_array.copy()
    if out.ndim == 2:                          # grayscale → RGB
        out = np.stack([out, out, out], axis=2).astype(np.uint8)
    elif out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    ys, xs = np.where(corners)
    H, W = out.shape[:2]
    for y, x in zip(ys, xs):
        # Draw a simple filled circle by checking distance
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        out[ny, nx] = color
    return out


def visualise_pipeline(img_array: np.ndarray,
                       name: str = "Image",
                       save_path: str = None,
                       **kwargs):
    """
    Full 6-panel figure showing every stage of the Harris pipeline.
    """
    gray    = rgb_to_grayscale(img_array)
    Ix, Iy  = sobel_gradients(gray)
    corners, R = detect_corners(gray, **kwargs)
    result  = draw_corners(img_array, corners)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Harris Corner Detector – {name}", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img_array, cmap='gray' if img_array.ndim == 2 else None)
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Grayscale")

    axes[0, 2].imshow(np.sqrt(Ix**2 + Iy**2), cmap='hot')
    axes[0, 2].set_title("Gradient Magnitude")

    r_display = R.copy(); r_display[r_display < 0] = 0
    axes[1, 0].imshow(r_display, cmap='hot')
    axes[1, 0].set_title("Corner Response R (positive)")

    axes[1, 1].imshow(corners, cmap='gray')
    axes[1, 1].set_title(f"Detected Corners ({corners.sum()} points)")

    axes[1, 2].imshow(result)
    axes[1, 2].set_title("Corners on Original Image")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    plt.show()
    return result


# ─────────────────────────────────────────────
#  5.  MAIN DEMO  (generates test images)
# ─────────────────────────────────────────────

def make_checkerboard(size: int = 256, squares: int = 8) -> np.ndarray:
    """Synthetic checkerboard – rich in corners."""
    sq = size // squares
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 255
    return board


def make_shapes(size: int = 256) -> np.ndarray:
    """Synthetic image with rectangle, triangle, circle – variety of features."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # light grey bg
    # Rectangle
    img[50:150, 50:180] = [70, 130, 180]
    # Triangle (filled via row-by-row)
    for row in range(170, 240):
        left  = int(128 - (row - 170) * 0.8)
        right = int(128 + (row - 170) * 0.8)
        img[row, max(0,left):min(size,right)] = [200, 80, 40]
    # Circle
    cx, cy, cr = 190, 80, 45
    y_idx, x_idx = np.ogrid[:size, :size]
    mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= cr**2
    img[mask] = [50, 180, 80]
    return img


def run_demo(image_path: str = None):
    """
    Run Harris detector on:
      1. Synthetic checkerboard
      2. Synthetic shapes
      3. A real image (if path given or downloaded)

    Output PNGs are saved next to this script file.
    """
    import os
    # Save outputs beside the script, wherever it lives
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Harris Corner Detector Demo")
    print("=" * 60)
    print(f"  Outputs will be saved to: {out_dir}")

    # ── Test 1: Checkerboard ──────────────────
    print("\n[1/3] Checkerboard (synthetic)")
    board = make_checkerboard(256, 8)
    visualise_pipeline(board, name="Checkerboard",
                       save_path=os.path.join(out_dir, "output_checkerboard.png"),
                       threshold_ratio=0.01)

    # ── Test 2: Shapes ────────────────────────
    print("\n[2/3] Geometric shapes (synthetic)")
    shapes = make_shapes(256)
    visualise_pipeline(shapes, name="Geometric Shapes",
                       save_path=os.path.join(out_dir, "output_shapes.png"),
                       threshold_ratio=0.01)

    # ── Test 3: Real image ────────────────────
    if image_path:
        print(f"\n[3/3] Real image: {image_path}")
        img = np.array(Image.open(image_path).convert('RGB'))
        visualise_pipeline(img, name="Real Image",
                           save_path=os.path.join(out_dir, "output_real.png"),
                           threshold_ratio=0.01)
    else:
        print("\n[3/3] Trying to download a public domain sample image …")
        try:
            import urllib.request
            url = "https://media.timeout.com/images/106106356/750/562/image.jpg"
            sample_path = os.path.join(out_dir, "sample.jpg")
            urllib.request.urlretrieve(url, sample_path)
            img = np.array(Image.open(sample_path).convert('RGB'))
            visualise_pipeline(img, name="Sample (Bikes)",
                               save_path=os.path.join(out_dir, "output_real.png"),
                               threshold_ratio=0.005)
        except Exception as e:
            print(f"  Could not download sample: {e}")
            print("  Skipping test 3. Pass your own image: python harris.py your_image.jpg")

    print("\nDone! Output images saved to:", out_dir)


# ─────────────────────────────────────────────
#  STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(img_path)
