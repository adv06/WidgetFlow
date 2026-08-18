import numpy as np


def smooth_score(val, scale, method="exp"):
    if method == "exp":
        return 100 * np.exp(-val / scale)
    elif method == "linear":
        return 100 * max(0.0, 1 - val / scale)
    elif method == "logistic":
        return 100 / (1 + np.exp(10 * (val - scale)))


def handling_layout(layout):
    MarginAsymmetry = smooth_score(layout["MarginAsymmetry"], 0.5, "exp")
    ContentAspectDiff = smooth_score(layout["ContentAspectDiff"], 0.05, "exp")
    AreaRatioDiff = smooth_score(layout["AreaRatioDiff"], 0.05, "exp")
    return {
        "MarginAsymmetry": round(MarginAsymmetry, 3),
        "ContentAspectDiff": round(ContentAspectDiff, 3),
        "AreaRatioDiff": round(AreaRatioDiff, 3),
    }


def handling_legibility(legibility):
    TextJaccard = 100 * np.clip(legibility.get("TextJaccard", 0), 0, 1)
    ContrastDiff = np.clip(legibility.get("ContrastDiff", 0), 0, 5)
    ContrastLocalDiff = np.clip(legibility.get("ContrastLocalDiff", 0), 0, 5)
    ContrastDiff = 100 * (1 - ContrastDiff / 5.0)
    ContrastLocalDiff = 100 * (1 - ContrastLocalDiff / 5.0)
    return {
        "TextJaccard": round(TextJaccard, 3),
        "ContrastDiff": round(ContrastDiff, 3),
        "ContrastLocalDiff": round(ContrastLocalDiff, 3),
    }


def handling_style(style):
    return {
        "PaletteDistance": round(100 * style.get("PaletteDistance"), 3),
        "Vibrancy": round(100 * style.get("Vibrancy"), 3),
        "PolarityConsistency": round(100 * style.get("PolarityConsistency"), 3),
    }


def handling_perceptual(perceptual):
    ssim = np.clip(perceptual.get("SSIM", 0), 0, 1)
    lp = np.clip(perceptual.get("LPIPS", 0), 0, 1)
    return {
        "ssim": round(ssim, 3),
        "lp": round(lp, 3),
    }


def composite_score(geo, perceptual, layout, legibility, style):
    """Organize metrics with transformations. Returns hierarchical dict."""
    _zero = lambda keys: {k: 0.0 for k in keys}

    layout_score = handling_layout(layout) if layout else _zero(["MarginAsymmetry", "ContentAspectDiff", "AreaRatioDiff"])
    legibility_score = handling_legibility(legibility) if legibility else _zero(["TextJaccard", "ContrastDiff", "ContrastLocalDiff"])
    style_score = handling_style(style) if style else _zero(["PaletteDistance", "Vibrancy", "PolarityConsistency"])
    perceptual_score = handling_perceptual(perceptual) if perceptual else _zero(["ssim", "lp"])
    geo_score = 100 * np.clip(geo, 0, 1) if geo is not None else 0.0

    return {
        "LayoutScore": layout_score,
        "LegibilityScore": legibility_score,
        "StyleScore": style_score,
        "PerceptualScore": perceptual_score,
        "Geometry": {"geo_score": float(geo_score)},
    }
