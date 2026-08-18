import numpy as np
import easyocr
import cv2

_reader = None


def _get_reader(gpu=True):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def to_gray(img):
    """Convert RGB image to grayscale float array [0,1]."""
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = img
    gray = gray.astype(np.float32)
    if gray.max() > 1:
        gray /= 255.0
    return np.clip(gray, 0, 1)


def contrast_ratio(img):
    """
    Approximate WCAG contrast ratio using 5-95 percentile luminance.
    """
    gray = to_gray(img)
    min_l, max_l = np.percentile(gray, [5, 95])
    return (max_l + 0.05) / (min_l + 0.05)


def ocr_text_easyocr(img, conf_thresh=0.5):
    """Extract visible text using EasyOCR."""
    reader = _get_reader()
    img_u8 = np.clip((img * 255).astype(np.uint8), 0, 255)
    results = reader.readtext(img_u8)
    words = [t for (_, t, conf) in results if conf >= conf_thresh and t.strip()]
    return " ".join(words), results


def local_contrast_from_text_regions(img, ocr_results, min_area=20):
    """Average contrast ratio within OCR-detected text regions."""
    gray = to_gray(img)
    H, W = gray.shape
    contrasts = []

    for (bbox, text, conf) in ocr_results:
        if conf < 0.5:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x_min, y_min = np.clip(np.min(pts[:, 0]), 0, W - 1), np.clip(np.min(pts[:, 1]), 0, H - 1)
        x_max, y_max = np.clip(np.max(pts[:, 0]), 0, W - 1), np.clip(np.max(pts[:, 1]), 0, H - 1)

        if (x_max - x_min) * (y_max - y_min) < min_area:
            continue

        patch = gray[y_min:y_max, x_min:x_max]
        if patch.size < 10:
            continue
        min_l, max_l = np.percentile(patch, [5, 95])
        contrasts.append((max_l + 0.05) / (min_l + 0.05))

    if len(contrasts) == 0:
        return None
    return float(np.mean(contrasts))


def compute_legibility(gt, gen):
    """
    Compute legibility metrics between GT and generated widget.

    Returns dict with: TextJaccard, ContrastDiff, ContrastLocalDiff
    """
    txt_gt, results_gt = ocr_text_easyocr(gt)
    txt_gen, results_gen = ocr_text_easyocr(gen)

    s_gt, s_gen = set(txt_gt.split()), set(txt_gen.split())
    jaccard = len(s_gt & s_gen) / (len(s_gt | s_gen) + 1e-6)

    contrast_gt = np.nan_to_num(contrast_ratio(gt))
    contrast_gen = np.nan_to_num(contrast_ratio(gen))
    contrast_diff = float(np.clip(abs(contrast_gt - contrast_gen), 0, 5))

    contrast_local_gt = local_contrast_from_text_regions(gt, results_gt)
    contrast_local_gen = local_contrast_from_text_regions(gen, results_gen)

    MAX_DIFF = 5.0
    if contrast_local_gt is not None and contrast_local_gen is not None:
        contrast_local_diff = abs(contrast_local_gt - contrast_local_gen)
    elif contrast_local_gt is None and contrast_local_gen is None:
        contrast_local_diff = 0.0
    else:
        contrast_local_diff = MAX_DIFF

    contrast_local_diff = float(np.clip(contrast_local_diff, 0, MAX_DIFF))

    return {
        "TextJaccard": float(jaccard),
        "ContrastDiff": contrast_diff,
        "ContrastLocalDiff": contrast_local_diff,
    }
