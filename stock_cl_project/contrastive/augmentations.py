import torch


def gaussian_jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """Add weak Gaussian noise to standardized time-series values."""
    if sigma <= 0:
        return x.clone()
    return x + torch.randn_like(x) * sigma


def random_value_mask(x: torch.Tensor, mask_ratio: float = 0.10) -> torch.Tensor:
    """Randomly replace values by zero (approximately feature mean after StandardScaler)."""
    if mask_ratio <= 0:
        return x.clone()
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1).")
    keep_mask = torch.rand_like(x) >= mask_ratio
    return x * keep_mask.to(x.dtype)


def make_contrastive_views(
    x: torch.Tensor,
    jitter_sigma: float = 0.01,
    mask_ratio: float = 0.10,
):
    """Create two conservative positive views with unchanged [B, L, N] shape.

    view1: jitter only
    view2: jitter + random value masking
    """
    view1 = gaussian_jitter(x, jitter_sigma)
    view2 = gaussian_jitter(x, jitter_sigma)
    view2 = random_value_mask(view2, mask_ratio)
    return view1, view2
