import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        return self.net(h)


class ContrastiveWrapper(nn.Module):
    """Temporary wrapper used only during CL pretraining.

    `backbone` is the exact forecasting model instance that will later be fine-tuned.
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        if not hasattr(backbone, "encode"):
            raise AttributeError("Backbone must expose encode(x, x_mark_enc=None, mask=None).")
        if not hasattr(backbone, "representation_dim"):
            raise AttributeError("Backbone must expose representation_dim.")
        self.backbone = backbone
        self.projector = ProjectionHead(backbone.representation_dim, hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, x_mark_enc=None):
        h = self.backbone.encode(x, x_mark_enc)
        if h.ndim != 2:
            raise RuntimeError(f"encode() must return [B,D], got {tuple(h.shape)}")
        z = self.projector(h)
        return F.normalize(z, dim=1)
