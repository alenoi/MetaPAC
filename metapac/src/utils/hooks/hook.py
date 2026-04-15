import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import pandas as pd
import torch


def _tensor_stats(x: torch.Tensor, prefix: str = "", include_quantiles: bool = True) -> Dict[str, Any]:
    """Per-tensor basic statistics optimized for small batches."""
    x = x.detach()
    numel = x.numel()
    if numel == 0:
        return {f"{prefix}numel": 0}

    # Most statistics can be computed on the GPU; only quantiles fall back to CPU if needed.
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    amin = x.amin().item()
    amax = x.amax().item()
    l2 = torch.linalg.vector_norm(x).item()
    l1 = torch.linalg.vector_norm(x, ord=1).item()

    # Sparsity: ratio of abs(x) < eps.
    eps = 1e-12 if x.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16) else 0
    sparsity = (x.abs() < eps).float().mean().item()

    stats = {
        f"{prefix}numel": numel,
        f"{prefix}mean": mean,
        f"{prefix}std": std,
        f"{prefix}min": amin,
        f"{prefix}max": amax,
        f"{prefix}l2": l2,
        f"{prefix}l1": l1,
        f"{prefix}sparsity": sparsity,
    }

    # Quantiles are optional because they can add noticeable overhead.
    if include_quantiles:
        try:
            q = torch.quantile(x.float(), torch.tensor([0.25, 0.5, 0.75], device=x.device))
            stats[f"{prefix}q25"] = q[0].item()
            stats[f"{prefix}q50"] = q[1].item()
            stats[f"{prefix}q75"] = q[2].item()
        except Exception:
            stats[f"{prefix}q25"] = float("nan")
            stats[f"{prefix}q50"] = float("nan")
            stats[f"{prefix}q75"] = float("nan")

    return stats


@dataclass
class Record:
    step: int
    phase: str  # "forward" or "backward"
    module: str
    shape: Tuple[int, ...]
    device: str
    dtype: str
    stats: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)


class HookManager:
    """
    Unified hook manager for forward and backward measurements.
    Usage:
        hm = HookManager()
        hm.register(module, "name", capture="both")
        ...
        with hm.capture():
            loss = model(**batch).loss
            loss.backward()
        df = hm.to_dataframe()
    """

    def __init__(
            self,
            reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            capture_grads_of: str = "output",  # "output" vagy "input"
            store_on_cpu: bool = False,
            keep_tensors: bool = False,
            capture_every_n_steps: int = 1,
            include_quantiles: bool = True,
    ):
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._records: List[Record] = []
        self._step: int = 0
        self.reduce_fn = reduce_fn
        self.capture_grads_of = capture_grads_of
        self.store_on_cpu = store_on_cpu
        self.keep_tensors = keep_tensors
        self.capture_every_n_steps = max(1, int(capture_every_n_steps))
        self.include_quantiles = bool(include_quantiles)

    def _should_capture_current_step(self) -> bool:
        return (self._step % self.capture_every_n_steps) == 0

    def clear(self):
        self._records.clear()
        self._step = 0

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.remove()

    def _pack_record(
            self,
            step: int,
            phase: str,
            module_name: str,
            tensor: torch.Tensor,
            extra_stats: Dict[str, Any],
    ):
        if self.store_on_cpu:
            tensor = tensor.detach().cpu()
        shape = tuple(tensor.shape)
        device = str(tensor.device)
        dtype = str(tensor.dtype)
        rec = Record(
            step=step,
            phase=phase,
            module=module_name,
            shape=shape,
            device=device,
            dtype=dtype,
            stats=extra_stats if not self.keep_tensors else {**extra_stats, "tensor": tensor},
        )
        self._records.append(rec)

    def _maybe_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduce_fn is None:
            return tensor
        try:
            return self.reduce_fn(tensor)
        except Exception:
            return tensor  # Fail-safe fallback.

    def register_parameters(self, module: torch.nn.Module, prefix: str = ""):
        """
        Register hooks for all named parameters (weights, biases) in a module.
        This provides parameter-level granularity instead of module-level.
        
        Args:
            module: The module whose parameters to register
            prefix: Naming prefix for the parameter (e.g., "model.layer.0")
        """
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{param_name}" if prefix else param_name

            # Register backward hook on the parameter's gradient
            if param.requires_grad:
                def make_param_hook(pname, p):
                    def param_grad_hook(grad):
                        if grad is not None and self._should_capture_current_step():
                            stats = _tensor_stats(grad, prefix="grad_", include_quantiles=self.include_quantiles)
                            # Add parameter statistics
                            param_stats = _tensor_stats(p.data, prefix="param_", include_quantiles=self.include_quantiles)
                            stats.update(param_stats)
                            self._pack_record(self._step, "parameter", pname, grad, stats)
                        return grad  # Don't modify gradient

                    return param_grad_hook

                handle = param.register_hook(make_param_hook(full_name, param))
                self._handles.append(handle)

    def register(self, module: torch.nn.Module, name: str, capture: str = "both"):
        """capture: 'forward', 'backward', or 'both'."""
        cap_fwd = capture in ("forward", "both")
        cap_bwd = capture in ("backward", "both")

        if cap_fwd:
            def fwd_hook(mod, inp, out):
                # out may be a tuple; consistently measure the first tensor.
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if not isinstance(tensor, torch.Tensor):
                    return
                if not self._should_capture_current_step():
                    return
                ten = self._maybe_reduce(tensor)
                stats = _tensor_stats(ten, prefix="act_", include_quantiles=self.include_quantiles)
                self._pack_record(self._step, "forward", name, ten, stats)

            self._handles.append(module.register_forward_hook(fwd_hook))

        if cap_bwd:
            # Use PyTorch 1.10+ full_backward_hook for backward capture.
            def bwd_hook(mod, gin, gout):
                # Choose from grad_output or grad_input.
                source = gout if self.capture_grads_of == "output" else gin
                if not isinstance(source, (tuple, list)) or len(source) == 0:
                    return
                tensor = source[0]
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    return
                if not self._should_capture_current_step():
                    return
                ten = self._maybe_reduce(tensor)
                stats = _tensor_stats(ten, prefix="actgrad_", include_quantiles=self.include_quantiles)
                # Use the same step that was advanced during the forward pass.
                self._pack_record(self._step, "backward", name, ten, stats)

            self._handles.append(module.register_full_backward_hook(bwd_hook))

    def next_step(self):
        self._step += 1

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._records:
            base = {
                "step": r.step,
                "phase": r.phase,
                "module": r.module,
                "shape": str(r.shape),
                "device": r.device,
                "dtype": r.dtype,
                "t": r.wall_time,
            }
            base.update(r.stats)
            rows.append(base)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def capture(self):
        """Context manager for step synchronization."""
        manager = self

        class _Ctx:
            def __enter__(self_inner):
                return manager

            def __exit__(self_inner, exc_type, exc, tb):
                manager.next_step()
                return False

        return _Ctx()
