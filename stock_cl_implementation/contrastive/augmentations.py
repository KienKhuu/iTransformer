import torch


class TimeSeriesAugmenter:
    """Small, semantics-preserving augmentations for standardized financial windows."""

    def __init__(self, jitter_std=0.02, mask_ratio=0.10):
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio

    def jitter(self, x):
        if self.jitter_std <= 0:
            return x
        return x + torch.randn_like(x) * self.jitter_std

    def mask(self, x):
        if self.mask_ratio <= 0:
            return x
        mask = torch.rand_like(x) < self.mask_ratio
        # The input is standardized, so zero corresponds approximately to the
        # training-set feature mean and is a neutral masking value.
        return x.masked_fill(mask, 0.0)

    def make_views(self, x):
        """
        Weak view  : small Gaussian jitter.
        Strong view: independent jitter + light element-wise masking.
        """
        view1 = self.jitter(x.clone())
        view2 = self.mask(self.jitter(x.clone()))
        return view1, view2
