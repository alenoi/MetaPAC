# metapac/src/compression/dequant.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch

# Never quantize / always restore from reference
_NEVER_QUANT_SUBSTR = (
    "layer_norm.weight", "layer_norm.bias",
    "LayerNorm.weight", "LayerNorm.bias",
)
_ALWAYS_RESTORE_IF_REF = _NEVER_QUANT_SUBSTR  # identical set


def _dq_affine(q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, axis: int) -> torch.Tensor:
    qf = q.to(torch.float32)
    if axis < 0 or q.ndim == 0:
        return (qf - zp.to(torch.float32)) * scale.to(torch.float32)
    # per-channel along axis
    shape = [1] * q.ndim
    shape[axis] = q.shape[axis]
    return (qf - zp.view(shape).to(torch.float32)) * scale.view(shape).to(torch.float32)


def _best_axis_dequant(
    q: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    meta_axis: int,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Try the requested axis first; if shapes mismatch, fall back gracefully."""
    if q.numel() == 0:
        return q.to(torch.float32)
    try_axes = [meta_axis]
    # robust fallbacks for common 2D weights
    if q.ndim == 2:
        try_axes += [1 - meta_axis, -1]
    else:
        try_axes += [-1]
    for ax in try_axes:
        try:
            out = _dq_affine(q, scale, zp, axis=ax)
            if baseline is not None and out.shape != baseline.shape:
                continue
            return out
        except Exception:
            continue
    # last resort: no per-channel broadcasting
    return (q.to(torch.float32) - zp.to(torch.float32)) * scale.to(torch.float32)


def dequant_state_dict(
    state: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    baseline_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Restore float32 weights in-place where metadata is available; skip safely otherwise."""
    for name, t in list(state.items()):
        if not isinstance(t, torch.Tensor):
            continue
        if any(s in name for s in _NEVER_QUANT_SUBSTR):
            # LN and similar remain in FP
            state[name] = t.to(torch.float32)
            continue

        meta_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None
        m = meta.get(name, None)
        if isinstance(m, dict) and {"scale", "zp", "axis"} <= set(m):
            meta_tuple = (m["scale"], m["zp"], int(m["axis"]))
        elif isinstance(m, (list, tuple)) and len(m) == 3:
            meta_tuple = (m[0], m[1], int(m[2]))

        if meta_tuple is None:
            # no metadata; if baseline has it and dtype is not FP, copy baseline
            if baseline_state is not None and name in baseline_state and t.dtype != torch.float32:
                state[name] = baseline_state[name].to(torch.float32)
            else:
                state[name] = t.to(torch.float32)
            continue

        scale, zp, axis = meta_tuple
        if not isinstance(scale, torch.Tensor) or not isinstance(zp, torch.Tensor):
            state[name] = t.to(torch.float32)
            continue
        if scale.numel() == 0:
            state[name] = t.to(torch.float32)
            continue

        base_w = None
        if baseline_state is not None and name in baseline_state:
            base_w = baseline_state[name].to(torch.float32)

        state[name] = _best_axis_dequant(t, scale, zp, meta_axis=axis, baseline=base_w)

    return state
