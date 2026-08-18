import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def load_image(path):
    """Load image as normalized RGB float array [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img) / 255.0


def to_gray(img):
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def lab_color_diff(img1, img2):
    """Mean and 95-percentile ΔE (CIE76)."""
    lab1, lab2 = rgb2lab(img1), rgb2lab(img2)
    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    return float(np.mean(diff)), float(np.percentile(diff, 95))


def edge_map(img):
    gray = to_gray(img)
    return cv2.Canny(gray, 100, 200)


def margin_from_mask(mask):
    """Return distances from content to edges (top, right, bottom, left)."""
    rows, cols = np.where(mask > 0)
    h, w = mask.shape
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, 0, 0]
    return [rows.min(), w - cols.max(), h - rows.max(), cols.min()]


def resize_to_match(gt, gen):
    """Resize generated image to GT size."""
    h_gt, w_gt = gt.shape[:2]
    gen_resized = cv2.resize(gen, (w_gt, h_gt), interpolation=cv2.INTER_AREA)
    return gen_resized


def remove_border_touching_components(mask):
    """
    mask: binary mask, 0/255 or 0/1
    returns cleaned binary mask
    """
    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    H, W = mask.shape
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]

        touches_border = (x == 0) or (y == 0) or (x + w == W) or (y + h == H)
        if not touches_border:
            cleaned[labels == i] = 255

    return cleaned.astype(np.uint8)