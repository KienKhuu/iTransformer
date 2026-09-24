import torch
import torch.nn as nn
import torch.nn.functional as F


def jitter(x, sigma=0.01):
    return x + sigma * torch.randn_like(x)


def temporal_mask(x, mask_ratio=0.10):
    if mask_ratio <= 0:
        return x
    mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) < mask_ratio
    return x.masked_fill(mask, 0.0)


def make_contrastive_views(x, jitter_sigma=0.01, mask_ratio=0.10):
    view1 = jitter(x, sigma=jitter_sigma)
    view2 = temporal_mask(jitter(x, sigma=jitter_sigma), mask_ratio=mask_ratio)
    return view1, view2


def info_nce_loss(z1, z2, temperature=0.2):
    """Symmetric cross-view InfoNCE with in-batch negatives."""
    if z1.size(0) < 2:
        return z1.new_zeros(())

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)


def _first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def infer_representation_layer(model_name, model):
    """Find a latent layer without editing original model source files."""
    modules = dict(model.named_modules())

    if model_name == "iTransformer" and "encoder" in modules:
        return "encoder"

    if model_name == "PatchTST":
        preferred = [
            "model.backbone",
            "backbone",
            "model.encoder",
            "encoder",
        ]
        for name in preferred:
            if name in modules:
                return name
        candidates = [
            name
            for name in modules
            if name and ("backbone" in name.lower() or "encoder" in name.lower())
        ]
        if candidates:
            return max(candidates, key=lambda name: name.count("."))

    if model_name == "LSTM":
        for name, module in model.named_modules():
            if isinstance(module, nn.LSTM):
                return name

    if model_name == "TSMixer":
        if hasattr(model, "model") and isinstance(model.model, nn.ModuleList):
            if len(model.model) > 0:
                return f"model.{len(model.model) - 1}"
        candidates = [
            name
            for name in modules
            if name and any(k in name.lower() for k in ["resblock", "mixing", "mixer"])
        ]
        if candidates:
            return max(candidates, key=lambda name: name.count("."))

    available = "\n".join(list(modules.keys())[:120])
    raise RuntimeError(
        f"Could not infer CL representation layer for {model_name}.\n"
        "Set the layer manually after inspecting named_modules().\n"
        f"Available module names (first 120):\n{available}"
    )


class ContrastiveWrapper(nn.Module):
    """External CL adapter around an unchanged forecasting backbone."""

    def __init__(
        self,
        backbone,
        model_name,
        projection_dim=64,
        projection_hidden=128,
        representation_layer=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.representation_layer = (
            representation_layer
            if representation_layer is not None
            else infer_representation_layer(model_name, backbone)
        )

        modules = dict(self.backbone.named_modules())
        if self.representation_layer not in modules:
            raise KeyError(
                f"Layer '{self.representation_layer}' not found in {model_name}."
            )

        self._captured = None
        self._hook_handle = modules[self.representation_layer].register_forward_hook(
            self._capture_hook
        )

        self.projector = nn.Sequential(
            nn.LazyLinear(projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, projection_dim),
        )

    def _capture_hook(self, module, inputs, output):
        self._captured = output

    def _latent_vector(self):
        h = _first_tensor(self._captured)
        if h is None:
            raise RuntimeError(
                f"Hook '{self.representation_layer}' returned no tensor for "
                f"{self.model_name}."
            )
        return h.reshape(h.size(0), -1)

    def forward(self, x, x_mark=None):
        self._captured = None
        output = self.backbone(x, x_mark)
        if self._captured is None:
            raise RuntimeError(
                f"No activation captured from '{self.representation_layer}' "
                f"for {self.model_name}."
            )
        return output

    def encode(self, x, x_mark=None):
        _ = self.forward(x, x_mark)
        h = self._latent_vector()
        return F.normalize(self.projector(h), dim=-1)

    def initialize_projection(self, x, x_mark=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self.encode(x, x_mark)
        self.train(was_training)

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
