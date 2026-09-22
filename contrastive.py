import torch
import torch.nn as nn
import torch.nn.functional as F


def jitter(x, sigma=0.01):
    """Small Gaussian perturbation in standardized feature space."""
    return x + sigma * torch.randn_like(x)


def temporal_mask(x, mask_ratio=0.10):
    """Mask complete timesteps (all variates) to zero = training-mean after scaling."""
    if mask_ratio <= 0:
        return x
    mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) < mask_ratio
    return x.masked_fill(mask, 0.0)


def make_contrastive_views(x, jitter_sigma=0.01, mask_ratio=0.10):
    # Weak view
    view1 = jitter(x, sigma=jitter_sigma)
    # Slightly stronger view
    view2 = temporal_mask(jitter(x, sigma=jitter_sigma), mask_ratio=mask_ratio)
    return view1, view2


def info_nce_loss(z1, z2, temperature=0.2):
    """Symmetric cross-view InfoNCE.

    Positive: z1[i] <-> z2[i]
    Negatives: all z2[j] (j != i) in the minibatch, and vice versa.
    """
    if z1.size(0) < 2:
        raise ValueError("InfoNCE requires batch_size >= 2")

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)


def _first_tensor(obj):
    """Extract the first tensor from tensor/tuple/list/dict hook output."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for item in obj.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def infer_representation_layer(model_name, model):
    """Infer a latent layer without modifying the original model source.

    The function intentionally fails loudly if it cannot find a sensible layer,
    instead of silently contrasting the final forecast output.
    """
    modules = dict(model.named_modules())

    if model_name == "iTransformer":
        if "encoder" in modules:
            return "encoder"

    elif model_name == "PatchTST":
        # THUML/official-style PatchTST typically exposes the TST encoder here.
        preferred = [
            "model.backbone",
            "backbone",
            "model.encoder",
            "encoder",
        ]
        for name in preferred:
            if name in modules:
                return name

        # Fallback: deepest module whose name contains 'backbone' or 'encoder'.
        candidates = [
            n for n in modules
            if n and ("backbone" in n.lower() or "encoder" in n.lower())
        ]
        if candidates:
            return max(candidates, key=lambda n: n.count("."))

    elif model_name == "LSTM":
        # Prefer the recurrent layer itself, before the forecast head.
        for name, module in model.named_modules():
            if isinstance(module, nn.LSTM):
                return name

    elif model_name == "TSMixer":
        # THUML TSMixer commonly stores ResBlocks inside a ModuleList named model.
        if hasattr(model, "model") and isinstance(model.model, nn.ModuleList):
            if len(model.model) > 0:
                return f"model.{len(model.model) - 1}"

        # Fallback: last residual/mixing block, avoiding the projection head.
        candidates = [
            n for n in modules
            if n and any(k in n.lower() for k in ["resblock", "mixing", "mixer"])
        ]
        if candidates:
            return max(candidates, key=lambda n: n.count("."))

    available = "\n".join(list(modules.keys())[:120])
    raise RuntimeError(
        f"Could not infer CL representation layer for {model_name}.\n"
        "Original model files were intentionally left unchanged.\n"
        "Set representation_layer explicitly after inspecting named_modules().\n"
        f"Available module names (first 120):\n{available}"
    )


class ContrastiveWrapper(nn.Module):
    """External CL adapter around an unchanged forecasting model.

    - The backbone object is the original model.
    - A forward hook captures an internal latent activation.
    - Only the projection head is newly added.
    - Gradients from InfoNCE flow through the captured activation back into the
      original backbone parameters.
    """

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

        # LazyLinear avoids hardcoding the latent size of four different models.
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
                f"Hook '{self.representation_layer}' did not return a tensor for "
                f"{self.model_name}."
            )
        if h.dim() < 2:
            raise RuntimeError(
                f"Captured representation has invalid shape {tuple(h.shape)}"
            )

        # Preserve all latent information without assuming architecture-specific
        # dimension ordering. The projection head learns the compression.
        return h.reshape(h.size(0), -1)

    def forward(self, x, x_mark=None):
        self._captured = None
        y = self.backbone(x, x_mark)
        if self._captured is None:
            raise RuntimeError(
                f"No activation captured from '{self.representation_layer}' in "
                f"{self.model_name}."
            )
        return y

    def encode(self, x, x_mark=None):
        # Running the unchanged backbone is enough for the hook to capture the
        # internal latent tensor. We intentionally ignore the forecast output.
        _ = self.forward(x, x_mark)
        h = self._latent_vector()
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def initialize_projection(self, x, x_mark=None):
        """Materialize LazyLinear before creating the optimizer."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self.encode(x, x_mark)
        self.train(was_training)

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
