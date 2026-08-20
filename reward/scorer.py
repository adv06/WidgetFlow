from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------- band model

BANDS: tuple[str, ...] = ("layout", "legibility", "style", "perceptual", "geometry")

# Which raw sub-metrics compose each band. Everything is mapped onto 0-100
# (higher = better) before aggregation.
BAND_PARTS: dict[str, tuple[str, ...]] = {
    "layout": ("MarginAsymmetry", "ContentAspectDiff", "AreaRatioDiff"),
    "legibility": ("TextJaccard", "ContrastDiff", "ContrastLocalDiff"),
    "style": ("PaletteDistance", "Vibrancy", "PolarityConsistency"),
    "perceptual": ("ssim", "lp"),
    "geometry": ("geo_score",),
}

# widget2code (base-Qwen pipeline) on the official bench — the bar to beat.
W2C_BAR: dict[str, float] = {
    "layout": 41.0, "legibility": 63.0, "style": 44.7,
    "perceptual": 67.8, "geometry": 100.0,
}


@dataclass
class BandScores:
    layout: float = 0.0
    legibility: float = 0.0
    style: float = 0.0
    perceptual: float = 0.0
    geometry: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __getitem__(self, k: str) -> float:
        return getattr(self, k)

    def mean(self) -> float:
        return float(np.mean([getattr(self, b) for b in BANDS]))

    def vs_bar(self) -> dict[str, float]:
        """Points above (positive) or below the widget2code bar, per band."""
        return {b: round(getattr(self, b) - W2C_BAR[b], 2) for b in BANDS}


# --------------------------------------------------------------- OCR caching

class OCRCache:
    """EasyOCR results for ground-truth images, keyed by path.

    The GT side of legibility is recomputed identically for every rollout of a
    widget and every pass over it; caching turns O(rollouts) OCR calls into one.
    The GENERATED side is still real OCR — substituting a proxy there is what
    made 3.0's in-loop legibility drift from the reported metric.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._mem: dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            try:
                self._mem = json.loads(self.path.read_text())
            except Exception:
                self._mem = {}

    def get(self, key: str):
        return self._mem.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._mem[key] = value

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._mem))
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._mem)


# -------------------------------------------------------------------- scorer

_lpips_ready = False


def _ensure_lpips(use_cuda: bool = True) -> None:
    """Load the LPIPS model once. Reloading per call dominates scoring time."""
    global _lpips_ready
    if _lpips_ready:
        return
    from widget_quality import perceptual as _p
    _p.set_device(use_cuda=use_cuda)
    _lpips_ready = True


def load_image(path: str | Path) -> np.ndarray:
    """RGB float array in [0,1] — the format every bench metric expects."""
    from widget_quality.utils import load_image as _li
    return _li(str(path))


def score(gt: np.ndarray, gen: np.ndarray, *,
          gt_key: str | None = None,
          ocr_cache: OCRCache | None = None,
          skip: Iterable[str] = (),
          use_cuda: bool = True) -> dict[str, float]:
    """The 11 official sub-metrics for one (gt, gen) pair.

    gt/gen are RGB float arrays in [0,1] (see load_image). Returns raw bench
    values on their native scales — call to_bands() to get 0-100 band numbers.

    gt_key + ocr_cache enable GT-side OCR caching. skip lets an expensive group
    be dropped ("legibility", "perceptual") when a caller only needs the cheap
    ones; skipped sub-metrics are simply absent from the result.
    """
    from widget_quality.geometry import compute_aspect_dimensionality_fidelity
    from widget_quality.layout import compute_layout
    from widget_quality.style import compute_style

    skip = set(skip)
    out: dict[str, float] = {}

    out.update({k: float(v) for k, v in compute_layout(gt, gen).items()})
    out.update({k: float(v) for k, v in compute_style(gt, gen).items()})
    out["geo_score"] = float(compute_aspect_dimensionality_fidelity(gt, gen))

    if "perceptual" not in skip:
        _ensure_lpips(use_cuda)
        from widget_quality.perceptual import compute_perceptual
        p = compute_perceptual(gt, gen)
        out["SSIM"] = float(p.get("SSIM", p.get("ssim", 0.0)))
        out["LPIPS"] = float(p.get("LPIPS", p.get("lp", 1.0)))

    if "legibility" not in skip:
        out.update({k: float(v) for k, v in
                    _legibility(gt, gen, gt_key, ocr_cache).items()})

    return out


def _legibility(gt, gen, gt_key: str | None, cache: OCRCache | None) -> dict[str, float]:
    """compute_legibility with the GT half cached.

    Mirrors widget_quality.legibility.compute_legibility exactly; the only
    change is reusing a stored GT OCR result instead of recomputing it.
    """
    from widget_quality.legibility import (
        contrast_ratio, local_contrast_from_text_regions, ocr_text_easyocr)

    # `cache is not None`, NOT `cache` — OCRCache defines __len__, so an
    # empty cache is falsy and would never populate itself.
    cached = cache.get(gt_key) if (cache is not None and gt_key) else None
    if cached is not None:
        txt_gt = cached.get("text", "")
        c_gt = float(cached.get("contrast", 0.0))
        cl_gt = cached.get("local_contrast")
    else:
        txt_gt, res_gt = ocr_text_easyocr(gt)
        c_gt = float(np.nan_to_num(contrast_ratio(gt)))
        cl_raw = local_contrast_from_text_regions(gt, res_gt)
        cl_gt = None if cl_raw is None else float(cl_raw)
        if cache is not None and gt_key:
            cache.put(gt_key, {"text": txt_gt, "contrast": c_gt, "local_contrast": cl_gt})

    txt_gen, res_gen = ocr_text_easyocr(gen)
    s_gt, s_gen = set(txt_gt.split()), set(txt_gen.split())
    jaccard = len(s_gt & s_gen) / (len(s_gt | s_gen) + 1e-6)

    c_gen = float(np.nan_to_num(contrast_ratio(gen)))
    contrast_diff = float(np.clip(abs(c_gt - c_gen), 0, 5))

    cl_raw = local_contrast_from_text_regions(gen, res_gen)
    cl_gen = None if cl_raw is None else float(cl_raw)
    MAX_DIFF = 5.0
    if cl_gt is not None and cl_gen is not None:
        cl_diff = abs(cl_gt - cl_gen)
    elif cl_gt is None and cl_gen is None:
        cl_diff = 0.0
    else:
        cl_diff = MAX_DIFF

    return {"TextJaccard": float(jaccard),
            "ContrastDiff": contrast_diff,
            "ContrastLocalDiff": float(np.clip(cl_diff, 0, MAX_DIFF))}


# ---------------------------------------------------------------- aggregation

def to_parts(raw: dict[str, float]) -> dict[str, float]:
    """Raw sub-metrics -> 0-100, higher-is-better, using the bench's own curves.

    This is `widget_quality.composite`, so the numbers match what the official
    CLI prints per sub-metric.
    """
    from widget_quality.composite import (
        handling_layout, handling_legibility, handling_perceptual, handling_style)

    parts: dict[str, float] = {}
    if all(k in raw for k in BAND_PARTS["layout"]):
        parts.update(handling_layout(raw))
    if all(k in raw for k in BAND_PARTS["style"]):
        parts.update(handling_style(raw))
    if all(k in raw for k in BAND_PARTS["legibility"]):
        parts.update(handling_legibility(raw))
    if "SSIM" in raw or "LPIPS" in raw:
        p = handling_perceptual(raw)
        # ssim: higher better. lp: LOWER better -> flip so the band is monotone.
        parts["ssim"] = 100.0 * float(p["ssim"])
        parts["lp"] = 100.0 * (1.0 - float(p["lp"]))
    if "geo_score" in raw:
        parts["geo_score"] = 100.0 * float(np.clip(raw["geo_score"], 0, 1))
    return parts


def to_bands(raw: dict[str, float]) -> BandScores:
    """5 band numbers from the raw sub-metrics.

    Each band is the unweighted MEAN of its 0-100 parts, matching 3.0 so
    numbers stay comparable across the two codebases.

    Known weakness, left visible rather than buried: an unweighted mean gives
    PolarityConsistency a third of the style band despite being the least
    stable of the three (it collapsed 59->35 in the corrector-v2 experiment
    while the other two improved). Reweighting is a one-line change here.
    """
    parts = to_parts(raw)
    out: dict[str, float] = {}
    for band, keys in BAND_PARTS.items():
        vals = [parts[k] for k in keys if k in parts]
        out[band] = float(np.mean(vals)) if vals else 0.0
    return BandScores(**out)


def score_bands(gt: np.ndarray, gen: np.ndarray, **kw) -> tuple[BandScores, dict[str, float]]:
    """Convenience: raw score + aggregation in one call. Returns (bands, raw)."""
    raw = score(gt, gen, **kw)
    return to_bands(raw), raw


# ------------------------------------------------------------------ regions

def crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop [x0,y0,x1,y1] (image pixels), clipped to bounds."""
    h, w = img.shape[:2]
    x0 = int(max(0, min(w - 1, round(box[0]))))
    y0 = int(max(0, min(h - 1, round(box[1]))))
    x1 = int(max(x0 + 1, min(w, round(box[2]))))
    y1 = int(max(y0 + 1, min(h, round(box[3]))))
    return img[y0:y1, x0:x1]


def score_region(gt: np.ndarray, gen: np.ndarray,
                 box: tuple[float, float, float, float],
                 gt_box: tuple[float, float, float, float] | None = None,
                 *, min_area: int = 400, **kw) -> dict[str, float] | None:
    """Score one region of both images. None if the crop is too small to trust.

    gt_box defaults to `box` — i.e. compare the same rectangle in both images.
    Note that comparing identical rectangles cannot detect POSITION error: a
    misplaced element is scored against whatever happens to sit at its own
    coordinates in the target. Pass a matched gt_box (from a search) when the
    caller has one.

    min_area exists because these metrics degrade on tiny crops — SSIM over a
    20x20 patch is mostly noise and OCR returns nothing at all.
    """
    g = crop(gen, box)
    t = crop(gt, gt_box if gt_box is not None else box)
    if g.size == 0 or t.size == 0:
        return None
    if g.shape[0] * g.shape[1] < min_area:
        return None
    if t.shape[:2] != g.shape[:2]:
        from PIL import Image
        t = np.asarray(Image.fromarray((t * 255).astype(np.uint8))
                       .resize((g.shape[1], g.shape[0]), Image.LANCZOS)) / 255.0
    return score(t, g, **kw)
