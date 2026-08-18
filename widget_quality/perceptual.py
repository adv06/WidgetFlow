import numpy as np
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim

_device = torch.device("cpu")
_lpips_vgg = None


def set_device(use_cuda=False):
    """Set device for LPIPS computation. Call before running evaluation."""
    global _device, _lpips_vgg

    if use_cuda and torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _lpips_vgg = LPIPS(net="vgg").to(_device)


def _ensure_model():
    global _lpips_vgg
    if _lpips_vgg is None:
        set_device(use_cuda=False)


def compute_perceptual(gt, gen):
    """
    Compute perceptual metrics.

    Returns dict with: SSIM, LPIPS
    """
    _ensure_model()

    ssim_val = ssim(gt, gen, channel_axis=2, data_range=1.0)

    gt_t = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0).float().to(_device)
    gen_t = torch.tensor(gen).permute(2, 0, 1).unsqueeze(0).float().to(_device)
    with torch.no_grad():
        lp = float(_lpips_vgg(gt_t, gen_t).item())

    return {
        "SSIM": ssim_val,
        "LPIPS": lp,
    }
