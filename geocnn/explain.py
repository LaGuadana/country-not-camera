"""Grad-CAM: which parts of the input pushed the model towards a class (Selvaraju et al. 2017).

On layer4 of this ResNet, which is followed only by global average pooling, dropout and one
linear layer, the Grad-CAM weights are the head's weights for the class divided by the number of
cells. The map is then the class activation map of Zhou et al. (2016): the class logit split over
the 4 x 6 feature-map cells (bias and ReLU aside). Earlier layers give finer but less faithful maps.
"""
import torch
import torch.nn.functional as F


def grad_cam(model, x, target=None, layer="layer4"):
    """x: a prepared (normalised) batch; the model must be in eval mode.
    Returns (maps, logits): maps (B, h, w) rescaled to [0, 1] per image, for `target`
    (default: each image's top class)."""
    store = {}

    def keep(module, inputs, out):
        out.retain_grad()
        store["a"] = out

    hook = getattr(model.backbone, layer).register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = model(x)
            target = logits.argmax(1) if target is None else target
            model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()
    finally:
        hook.remove()
    a = store["a"]
    cam = F.relu((a.grad.mean((2, 3), keepdim=True) * a).sum(1)).detach()
    lo, hi = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    return (cam - lo) / (hi - lo).clamp_min(1e-8), logits.detach()
