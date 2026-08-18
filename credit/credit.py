from __future__ import annotations

import tempfile
from pathlib import Path

from generator.render_jsx import JSXRenderer
from widget_quality.utils import load_image

from .ablate import credit_tokens
from .resolve import resolve_dead, dead_value_map

BANDS = ("layout", "style", "perceptual")
WEIGHTS = {"layout": 0.30, "style": 0.30, "perceptual": 0.20, "legibility": 0.20}


class BandScorer:

    def __init__(self, gt_path: str, *, skip=("legibility",), use_cuda=False,
                 scorer=None):
        self.gt_path = str(gt_path)
        self.gt = load_image(self.gt_path)
        self.h, self.w = self.gt.shape[:2]
        self.skip = tuple(skip)
        self.use_cuda = use_cuda
        self._r = None
        self._tmp = Path(tempfile.mkdtemp(prefix="wfcredit_"))
        self.n_renders = 0
        self._score_bands = scorer or self._default_scorer()

    @staticmethod
    def _default_scorer():
        try:
            from reward.scorer import score_bands
            return score_bands
        except Exception as exc:
            raise RuntimeError(
                "no band scorer available; pass scorer=... explicitly") from exc

    def __enter__(self):
        self._r = JSXRenderer().__enter__()
        return self

    def __exit__(self, *exc):
        if self._r is not None:
            self._r.__exit__(*exc)
            self._r = None
        return False


    def __call__(self, code: str):
        if self._r is None:
            raise RuntimeError("use BandScorer as a context manager")
        png = self._tmp / f"r{self.n_renders % 4}.png"
        self.n_renders += 1
        try:
            self._r.render(code, "", self.w, self.h, png)
            bands, _ = self._score_bands(self.gt, load_image(str(png)),
                                         use_cuda=self.use_cuda, skip=self.skip)


            bd = bands.as_dict() if hasattr(bands, "as_dict") else bands
            return {b: float(bd[b]) for b in BANDS if b in bd}
        except Exception:
            return None

    def trace(self, code: str):
        png = self._tmp / "base.png"
        return self._r.render_with_trace(jsx=code, css="", width=self.w,
                                         height=self.h, out_path=png)


def spans(trace) -> list[dict]:
    out = []
    for e in (trace or {}).get("elements", []):
        if (e.get("display") == "none" or e.get("visibility") == "hidden"
                or float(e.get("opacity", 1) or 1) <= 0.01):
            continue
        try:
            a, b = str(e.get("src", "")).split(":", 1)
            a, b = int(a), int(b)
        except Exception:
            continue
        if b > a >= 0:
            out.append({"idx": e.get("idx"), "tag": e.get("tag"),
                        "start": a, "end": b})
    out.sort(key=lambda d: d["start"])
    return out


def credit_widget(code: str, scorer: BandScorer, *, band_weights=None,
                  min_delta=0.5, max_tokens=None, prefilter=True, verbose=False):

    dead = {}
    if prefilter:
        res = resolve_dead(scorer._r, code, scorer.w, scorer.h)
        dead = dead_value_map(res)
        elements = spans(res["trace"])
    else:
        elements = spans(scorer.trace(code))
    return credit_tokens(code, elements, scorer,
                         band_weights=band_weights or WEIGHTS,
                         min_delta=min_delta, max_tokens=max_tokens,
                         dead=dead, verbose=verbose)
