"""Widget Quality — evaluation toolkit for widget generation quality."""

__version__ = "0.1.0"

from .composite import composite_score
from .geometry import compute_aspect_dimensionality_fidelity
from .layout import compute_layout
from .style import compute_style
from .utils import load_image, resize_to_match

# Eval-only extras with heavy deps (easyocr/lpips). Guarded so the
# geometry/layout reward path (used by DAPO) still imports when those
# optional deps are absent.
try:
    from .legibility import compute_legibility
    from .perceptual import compute_perceptual, set_device
except ImportError:
    compute_legibility = None  # type: ignore[assignment]
    compute_perceptual = None  # type: ignore[assignment]
    set_device = None  # type: ignore[assignment]

__all__ = [
    "composite_score",
    "compute_aspect_dimensionality_fidelity",
    "compute_layout",
    "compute_legibility",
    "compute_perceptual",
    "compute_style",
    "set_device",
    "load_image",
    "resize_to_match",
]
