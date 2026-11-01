"""Inference loop shared by training-time validation and scripts/evaluate.py."""
import numpy as np
import torch


def amp_dtype(device):
    if device.type != "cuda":
        return None
    # is_bf16_supported() also counts emulated bf16 (T4, P100); only use it with hardware support
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


@torch.inference_mode()
def predict(model, loader, device, prep, features=False, channels_last=False):
    """`prep` turns the loader's uint8 batches into normalised model input (geocnn.data.Prep).
    Returns logits (n, K) float32, labels (n,), dataset indices (n,) and, if asked, the
    512-d pooled features (n, 512)."""
    model.eval()
    dtype = amp_dtype(device)
    out_logits, out_labels, out_idx, out_feats = [], [], [], []
    for x, y, idx in loader:
        x = prep(x.to(device, non_blocking=True))
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits, f = model(x, return_features=True)
        out_logits.append(logits.float().cpu().numpy())
        out_labels.append(y.numpy())
        out_idx.append(idx.numpy())
        if features:
            out_feats.append(f.float().cpu().numpy())
    res = {"logits": np.concatenate(out_logits), "labels": np.concatenate(out_labels),
           "idx": np.concatenate(out_idx)}
    if features:
        res["features"] = np.concatenate(out_feats)
    return res
