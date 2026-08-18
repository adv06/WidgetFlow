import cv2
import numpy as np
from scipy.spatial.distance import cdist

from .utils import edge_map, margin_from_mask, remove_border_touching_components

MAX_DIFF = 5.0


def compute_margin_asymmetry(mask_gt, mask_gen):
    """Variance imbalance of margins (normalized by mean)."""
    m_gt = margin_from_mask(mask_gt)
    m_gen = margin_from_mask(mask_gen)
    diffs = np.abs(np.array(m_gt) - np.array(m_gen))
    mean = np.mean(diffs)
    return 0.0 if mean < 1e-6 else float(np.std(diffs) / mean)


def compute_content_aspect_diff(mask_gt, mask_gen):
    """Difference in content bounding-box aspect ratio."""
    if np.sum(mask_gt) == 0 or np.sum(mask_gen) == 0:
        return MAX_DIFF

    def bbox_ar(mask):
        ys, xs = np.where(mask > 0)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        return w / h if h > 0 else 1.0

    ar_gt, ar_gen = bbox_ar(mask_gt), bbox_ar(mask_gen)
    return float(abs(np.log(ar_gt / ar_gen)))


def analyze_internal_structure(mask_gt, mask_gen, min_area=10):
    """
    Compare internal content structure (connected components).
    Returns element area ratio difference.
    """
    def get_components(mask):
        mask_bin = (mask > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        stats = stats[1:]  # skip background
        boxes = [(x, y, w, h, w * h) for x, y, w, h, area in stats if area > min_area]
        return boxes

    boxes_gt = get_components(mask_gt)
    boxes_gen = get_components(mask_gen)

    areas_gt = np.array([b[4] for b in boxes_gt])
    areas_gen = np.array([b[4] for b in boxes_gen])
    if len(areas_gt) > 0 and len(areas_gen) > 0:
        area_ratio_diff = abs(
            (areas_gen.mean() / areas_gen.sum()) - (areas_gt.mean() / areas_gt.sum())
        )
    else:
        area_ratio_diff = MAX_DIFF

    return {"AreaRatioDiff": float(area_ratio_diff)}


def compute_layout(gt, gen):
    """
    Compute layout metrics between GT and generated widget.

    Returns dict with: MarginAsymmetry, ContentAspectDiff, AreaRatioDiff
    """
    e_gt, e_gen = edge_map(gt), edge_map(gen)
    kernel = np.ones((3, 3), np.uint8)
    mask_gt = cv2.dilate(e_gt, kernel)
    mask_gen = cv2.dilate(e_gen, kernel)

    mask_gt = remove_border_touching_components(mask_gt)
    mask_gen = remove_border_touching_components(mask_gen)

    margin_asym = compute_margin_asymmetry(mask_gt, mask_gen)
    aspect_diff = compute_content_aspect_diff(mask_gt, mask_gen)
    inner = analyze_internal_structure(mask_gt, mask_gen)

    return {
        "MarginAsymmetry": float(margin_asym),
        "ContentAspectDiff": float(aspect_diff),
        "AreaRatioDiff": inner["AreaRatioDiff"],
    }
