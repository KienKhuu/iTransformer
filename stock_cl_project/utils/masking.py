import torch


class TriangularCausalMask:
    """Minimal causal mask compatible with the attention layer used by this project."""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self):
        return self._mask
