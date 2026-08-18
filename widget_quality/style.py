import numpy as np
import cv2
from skimage.color import rgb2hsv, rgb2lab, rgb2gray
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment


def compute_palette_distance(gt, gen, bins=36):
    """Hue histogram Earth-Mover's Distance."""
    hsv_gt, hsv_gen = rgb2hsv(gt), rgb2hsv(gen)
    h_gt, h_gen = hsv_gt[..., 0].ravel(), hsv_gen[..., 0].ravel()

    hist_gt, _ = np.histogram(h_gt, bins=bins, range=(0, 1), density=True)
    hist_gen, _ = np.histogram(h_gen, bins=bins, range=(0, 1), density=True)

    emd = wasserstein_distance(
        np.arange(bins), np.arange(bins),
        hist_gt / (hist_gt.sum() + 1e-6),
        hist_gen / (hist_gen.sum() + 1e-6),
    )
    score = float(np.exp(-emd / (bins * 0.08)))
    return np.clip(score, 0, 1)


def compute_vibrancy_consistency(gt, gen, bins=30):
    """HSV saturation histogram EMD."""
    hsv_gt, hsv_gen = rgb2hsv(gt), rgb2hsv(gen)
    s_gt, s_gen = hsv_gt[..., 1].ravel(), hsv_gen[..., 1].ravel()
    hist_gt, _ = np.histogram(s_gt, bins=bins, range=(0, 1), density=True)
    hist_gen, _ = np.histogram(s_gen, bins=bins, range=(0, 1), density=True)
    emd = wasserstein_distance(
        np.arange(bins), np.arange(bins),
        hist_gt / (hist_gt.sum() + 1e-6),
        hist_gen / (hist_gen.sum() + 1e-6),
    )
    score = float(np.exp(-emd / (bins * 0.05)))
    return np.clip(score, 0, 1)


def compute_polarity_consistency(gt, gen, q=0.1, eps=1e-6):
    L_gt = rgb2gray(gt)
    L_gen = rgb2gray(gen)

    def get_polarity_stats(L):
        flat = np.sort(L.ravel())
        k = max(1, int(q * flat.size))

        bg = np.median(flat)
        dark = np.mean(flat[:k])
        bright = np.mean(flat[-k:])

        # choose the stronger contrast side relative to bg
        if abs(bg - dark) >= abs(bg - bright):
            fg = dark
        else:
            fg = bright

        contrast = bg - fg
        polarity = np.sign(contrast)
        strength = abs(contrast)
        return polarity, strength

    pol_gt, str_gt = get_polarity_stats(L_gt)
    pol_gen, str_gen = get_polarity_stats(L_gen)

    # reject nearly flat images
    if str_gt < eps or str_gen < eps:
        return 0.0

    pol_score = 1.0 if pol_gt == pol_gen else 0.0
    mag_diff = abs(str_gt - str_gen)

    score = pol_score * np.exp(-mag_diff * 5)
    return float(np.clip(score, 0, 1))


def compute_style(gt, gen):
    """
    Compute style metrics.

    Returns dict with: PaletteDistance, Vibrancy, PolarityConsistency
    """
    return {
        "PaletteDistance": compute_palette_distance(gt, gen),
        "Vibrancy": compute_vibrancy_consistency(gt, gen),
        "PolarityConsistency": compute_polarity_consistency(gt, gen),
    }
