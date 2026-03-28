"""
=============================================================
augmentation.py  —  Image Augmentation for Training
=============================================================

WHY DO WE NEED THIS?
--------------------
The IAM dataset has a limited number of images. If we train only on
those exact images, the model will MEMORIZE them instead of LEARNING
general handwriting patterns. This is called OVERFITTING.

Augmentation solves this by creating slightly modified versions of
each image on the fly during training:
  - Slightly tilted?   → model learns rotation-invariance
  - More noise?        → model learns to handle noisy scans
  - Slightly stretched? → model handles different writing speeds

IMPORTANT RULE: Augmentation is applied ONLY during training.
During validation and inference, use the ORIGINAL clean image.
Augmenting test data would give unfair/incorrect results.

Each function returns a modified copy — the original is never changed.
=============================================================
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Add random Gaussian noise to the image.

    WHY: Real-world scanned documents often have sensor noise or
    paper grain. Training with noise makes the model more robust
    when deployed on real-world scanner/camera input.

    HOW IT WORKS:
    Gaussian noise follows a normal distribution N(0, sigma).
    We add random values from this distribution to each pixel.

    Parameters
    ----------
    img   : np.ndarray  — preprocessed float32 image (H, W, 1), values [0,1]
    sigma : float       — noise intensity. 0.05 = mild, 0.2 = heavy

    Returns
    -------
    np.ndarray  — noisy image, clipped to [0.0, 1.0]
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img + noise
    # Clip to keep values in valid range [0, 1]
    return np.clip(noisy, 0.0, 1.0)


def random_rotation(img: np.ndarray, max_angle: float = 3.0) -> np.ndarray:
    """
    Randomly rotate the image by a small angle.

    WHY: Handwriters naturally write at slightly different angles.
    A few degrees rotation teaches the model to handle slightly
    tilted or slanted text (common in real handwriting).

    We keep the rotation SMALL (±3°) because large rotations would
    change the text layout too much and make labels incorrect.

    HOW IT WORKS:
    Rotation is applied around the image center using an affine warp matrix.
    Empty regions (from rotation) are filled with white (background color).

    Parameters
    ----------
    img       : np.ndarray  — float32 image (H, W, 1), values [0,1]
    max_angle : float       — maximum rotation in degrees (±max_angle)

    Returns
    -------
    np.ndarray  — rotated image, same shape as input
    """
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]

    # Build the rotation matrix around the image center
    center   = (w / 2.0, h / 2.0)
    rot_mat  = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Apply the rotation; fill new pixels with white (0.0 after our inversion)
    rotated = cv2.warpAffine(
        img[:, :, 0],          # remove channel dim for warpAffine
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0        # 0.0 = background (remember we inverted pixels)
    )

    return np.expand_dims(rotated, axis=-1)


def random_erosion_dilation(img: np.ndarray) -> np.ndarray:
    """
    Randomly apply erosion or dilation to simulate pen thickness variation.

    WHY:
    - Erosion  : makes strokes THINNER  → simulates light pen pressure
    - Dilation : makes strokes THICKER  → simulates heavy pen pressure / bold ink

    Training with both teaches the model to recognize the same character
    regardless of how thick or thin the ink strokes are.

    Parameters
    ----------
    img : np.ndarray  — float32 image (H, W, 1), values [0,1]

    Returns
    -------
    np.ndarray  — morphologically modified image
    """
    # Small 2x2 structuring element (kernel) defines the neighborhood
    kernel = np.ones((2, 2), dtype=np.uint8)

    # Randomly pick erosion or dilation with equal probability
    op = np.random.choice(["erode", "dilate"])

    img_uint8 = (img[:, :, 0] * 255).astype(np.uint8)

    if op == "erode":
        result = cv2.erode(img_uint8, kernel, iterations=1)
    else:
        result = cv2.dilate(img_uint8, kernel, iterations=1)

    result_float = result.astype(np.float32) / 255.0
    return np.expand_dims(result_float, axis=-1)


def random_blur(img: np.ndarray, max_sigma: float = 0.8) -> np.ndarray:
    """
    Apply random Gaussian blur to simulate out-of-focus or motion blur.

    WHY: Camera-captured documents are often slightly blurry.
    Training with blurred images makes the model handle real-world inputs.

    Parameters
    ----------
    img       : np.ndarray  — float32 image (H, W, 1)
    max_sigma : float       — maximum blur sigma (0 = no blur, 1 = moderate)

    Returns
    -------
    np.ndarray  — blurred image
    """
    sigma = np.random.uniform(0, max_sigma)
    blurred = gaussian_filter(img[:, :, 0], sigma=sigma)
    return np.expand_dims(blurred.astype(np.float32), axis=-1)


def random_brightness(img: np.ndarray, delta: float = 0.15) -> np.ndarray:
    """
    Randomly change the brightness of the image.

    WHY: Scanned documents vary in brightness depending on the scanner settings,
    age of paper, and ink density. Brightness variation makes the model
    handle a wider range of real document qualities.

    Parameters
    ----------
    img   : np.ndarray  — float32 image (H, W, 1), values [0,1]
    delta : float       — maximum brightness shift (±delta)

    Returns
    -------
    np.ndarray  — brightness-shifted image, clipped to [0.0, 1.0]
    """
    shift = np.random.uniform(-delta, delta)
    return np.clip(img + shift, 0.0, 1.0)


def elastic_distortion(img: np.ndarray, alpha: float = 8.0, sigma: float = 3.0) -> np.ndarray:
    """
    Apply elastic deformation to simulate natural handwriting variability.

    WHY: Every person writes the same letter slightly differently each time.
    Elastic distortion randomly warps the image in a smooth, organic way —
    mimicking the natural variation in human penmanship.

    This is one of the most powerful augmentations for handwriting recognition.

    HOW IT WORKS:
    1. Generate two random displacement fields (dx, dy) — one per axis
    2. Smooth them with Gaussian blur to make the displacement smooth
    3. Apply the fields to pixel coordinates using cv2.remap

    Parameters
    ----------
    img   : np.ndarray  — float32 image (H, W, 1)
    alpha : float       — intensity of displacement (higher = more distortion)
    sigma : float       — smoothness of displacement (higher = smoother warping)

    Returns
    -------
    np.ndarray  — elastically distorted image
    """
    h, w = img.shape[:2]

    # Random displacement fields in x and y directions
    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha

    # Build the remapping coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Remap pixels using the distorted coordinate grid
    distorted = cv2.remap(
        img[:, :, 0], map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return np.expand_dims(distorted.astype(np.float32), axis=-1)


def augment_image(img: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
    """
    Apply a random combination of augmentations to a single image.

    WHY: Combining multiple augmentations creates far more diverse training
    samples. Each call to this function produces a slightly different version
    of the image — effectively multiplying your dataset size.

    Each individual augmentation is applied with probability `augment_prob`,
    so not every augmentation fires every time. This creates variety.

    Parameters
    ----------
    img          : np.ndarray  — preprocessed float32 image (H, W, 1)
    augment_prob : float       — probability [0,1] to apply each augmentation

    Returns
    -------
    np.ndarray  — augmented image (same shape as input)
    """
    aug = img.copy()

    # Each augmentation fires independently with the set probability
    if np.random.random() < augment_prob:
        aug = add_gaussian_noise(aug, sigma=0.03)

    if np.random.random() < augment_prob:
        aug = random_rotation(aug, max_angle=2.5)

    if np.random.random() < augment_prob:
        aug = random_brightness(aug, delta=0.12)

    if np.random.random() < augment_prob:
        aug = random_blur(aug, max_sigma=0.6)

    # Elastic distortion is very powerful — apply with lower probability
    if np.random.random() < (augment_prob * 0.5):
        aug = elastic_distortion(aug, alpha=6.0, sigma=3.0)

    # Erosion/dilation fires rarely — extreme stroke changes can hurt readability
    if np.random.random() < (augment_prob * 0.3):
        aug = random_erosion_dilation(aug)

    return aug


def simulate_low_contrast(img: np.ndarray,
                          min_range: float = 0.15,
                          max_range: float = 0.5) -> np.ndarray:
    """
    Compress dynamic range to simulate low-contrast camera photos.

    Camera-captured document images typically have a narrow pixel range
    (e.g., 0.4–0.7 instead of 0.0–1.0). This augmentation compresses
    the clean training image into a random narrow band, forcing the model
    to learn to read text regardless of absolute contrast.

    Parameters
    ----------
    img       : np.ndarray  — float32 image (H, W, 1), values [0,1]
    min_range : float       — minimum width of the compressed range
    max_range : float       — maximum width of the compressed range

    Returns
    -------
    np.ndarray  — contrast-compressed image, values still in [0,1]
    """
    range_width = np.random.uniform(min_range, max_range)
    low = np.random.uniform(0.0, 1.0 - range_width)
    high = low + range_width
    return (img * (high - low) + low).astype(np.float32)


def simulate_uneven_lighting(img: np.ndarray,
                             strength: float = 0.3) -> np.ndarray:
    """
    Add a smooth gradient to simulate uneven illumination from a camera.

    Phone cameras often produce images that are brighter on one side
    (near the light source) and darker on the other.

    Parameters
    ----------
    img      : np.ndarray  — float32 image (H, W, 1), values [0,1]
    strength : float       — maximum brightness shift at image edges

    Returns
    -------
    np.ndarray  — unevenly lit image, clipped to [0,1]
    """
    h, w = img.shape[:2]
    # Random gradient direction
    direction = np.random.choice(["horizontal", "vertical", "diagonal"])
    s = np.random.uniform(0.1, strength)

    if direction == "horizontal":
        gradient = np.linspace(-s, s, w).reshape(1, w, 1)
    elif direction == "vertical":
        gradient = np.linspace(-s, s, h).reshape(h, 1, 1)
    else:
        gx = np.linspace(-s, s, w).reshape(1, w)
        gy = np.linspace(-s, s, h).reshape(h, 1)
        gradient = (gx + gy).reshape(h, w, 1) / 2.0

    return np.clip(img + gradient, 0.0, 1.0).astype(np.float32)


def simulate_camera_noise(img: np.ndarray) -> np.ndarray:
    """
    Add salt-and-pepper style noise to simulate camera sensor artifacts.

    Parameters
    ----------
    img : np.ndarray  — float32 image (H, W, 1), values [0,1]

    Returns
    -------
    np.ndarray  — noisy image
    """
    noise_density = np.random.uniform(0.005, 0.02)
    mask = np.random.random(img.shape)
    out = img.copy()
    out[mask < noise_density / 2] = 0.0
    out[mask > 1.0 - noise_density / 2] = 1.0
    return out.astype(np.float32)


def augment_image_camera(img: np.ndarray,
                         augment_prob: float = 0.5) -> np.ndarray:
    """
    Apply augmentations that simulate camera-captured document images.

    This is a superset of augment_image() — it includes all the original
    augmentations PLUS camera-specific effects: low contrast, uneven
    lighting, and salt-pepper noise.

    Use this for fine-tuning the model to handle camera photos.

    Parameters
    ----------
    img          : np.ndarray  — preprocessed float32 image (H, W, 1)
    augment_prob : float       — probability [0,1] to apply each augmentation

    Returns
    -------
    np.ndarray  — augmented image (same shape as input)
    """
    aug = img.copy()

    # Standard augmentations (always applicable)
    if np.random.random() < augment_prob:
        aug = add_gaussian_noise(aug, sigma=np.random.uniform(0.02, 0.06))

    if np.random.random() < augment_prob:
        aug = random_rotation(aug, max_angle=3.0)

    if np.random.random() < augment_prob:
        aug = random_brightness(aug, delta=0.2)

    if np.random.random() < augment_prob:
        aug = random_blur(aug, max_sigma=1.0)

    if np.random.random() < (augment_prob * 0.5):
        aug = elastic_distortion(aug, alpha=8.0, sigma=3.0)

    if np.random.random() < (augment_prob * 0.3):
        aug = random_erosion_dilation(aug)

    # Camera-specific augmentations (higher probability to ensure model sees them)
    if np.random.random() < 0.6:
        aug = simulate_low_contrast(aug, min_range=0.2, max_range=0.55)

    if np.random.random() < 0.5:
        aug = simulate_uneven_lighting(aug, strength=0.25)

    if np.random.random() < 0.3:
        aug = simulate_camera_noise(aug)

    return aug


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    # Create a dummy float32 image to test all augmentations
    dummy = np.random.rand(64, 512, 1).astype(np.float32)

    tests = [
        ("Gaussian Noise",    add_gaussian_noise(dummy)),
        ("Random Rotation",   random_rotation(dummy)),
        ("Brightness",        random_brightness(dummy)),
        ("Blur",              random_blur(dummy)),
        ("Elastic Distort",   elastic_distortion(dummy)),
        ("Erosion/Dilation",  random_erosion_dilation(dummy)),
        ("Full augment_image", augment_image(dummy)),
    ]

    print("Augmentation Self-Test")
    print("-" * 50)
    for name, result in tests:
        ok = (result.shape == dummy.shape and
              result.min() >= 0.0 and
              result.max() <= 1.0)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<25} shape={result.shape}")
