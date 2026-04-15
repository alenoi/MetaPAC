# metapac/src/compression/variable_bit_layers.py
# Robust shims for variable-bit export: safe kwargs filtering, None-handling,
# registry, meta iteration, memory accounting, replacement utility,
# and in-place FP32 loaders with transpose-aware copying.

from __future__ import annotations

from typing import Iterable, Dict, Any, Generator, Optional, List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "QuantizedLinear",
    "QuantizedEmbedding",
    "ensure_registry",
    "register_quantized_layer",
    "iter_quant_meta_from_model",
    "calculate_memory_savings",
    "replace_linear_with_quantized",
    "_iter_quant_meta",
]


# ----------------------------- helpers -----------------------------
def _pop_quant_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    qkeys = [
        "bits", "per_channel", "per_token", "group_size",
        "scale", "scales", "zero_point", "zero_points",
        "quant_axis", "axis", "dtype_q", "rounding", "observer",
        "fake_quant", "symmetric", "asymmetric",
    ]
    q = {}
    for k in list(kwargs.keys()):
        if k in qkeys:
            q[k] = kwargs.pop(k)
    return q


def _safe_int_or_default(x, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _safe_bool_or_default(x, default: bool) -> bool:
    if x is None:
        return default
    try:
        return bool(x)
    except Exception:
        return default


def _register_optional_buffer(mod: nn.Module, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        mod.register_buffer(name, value, persistent=True)
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        try:
            stacked = torch.stack(list(value))
            mod.register_buffer(name, stacked, persistent=True)
        except Exception:
            mod.register_buffer(name, value[0], persistent=True)
            setattr(mod, name + "_list", value)
    else:
        setattr(mod, name, value)


def _update_meta_shape_numel(mod: nn.Module) -> None:
    if hasattr(mod, "quant_meta") and hasattr(mod, "weight") and mod.weight is not None:
        mod.quant_meta["weight_numel"] = int(mod.weight.numel())
        mod.quant_meta["shape"] = tuple(mod.weight.shape)


# ========================== Shim: QuantizedLinear ==========================
class QuantizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        q = _pop_quant_kwargs(kwargs)
        super().__init__(in_features, out_features, bias=bias, **kwargs)

        self.bits: int = _safe_int_or_default(q.get("bits", 8), 8)
        self.per_channel: bool = _safe_bool_or_default(q.get("per_channel", True), True)
        self.group_size: Optional[int] = q.get("group_size", None)

        _register_optional_buffer(self, "qscale", q.get("scale"))
        _register_optional_buffer(self, "qzero_point", q.get("zero_point"))
        _register_optional_buffer(self, "qscales", q.get("scales"))
        _register_optional_buffer(self, "qzero_points", q.get("zero_points"))

        self.quant_meta = {
            "name": None,
            "bits": self.bits,
            "weight_numel": int(self.weight.numel()),
            "shape": tuple(self.weight.shape),
        }

    @classmethod
    def from_linear(
            cls,
            linear: nn.Linear,
            bits: int = 8,
            per_channel: bool = True,
            **kwargs,
    ) -> "QuantizedLinear":
        qlayer = cls(
            linear.in_features, linear.out_features,
            bias=(linear.bias is not None),
            bits=bits, per_channel=per_channel, **kwargs
        )
        with torch.no_grad():
            qlayer.weight.copy_(linear.weight)
            if linear.bias is not None and qlayer.bias is not None:
                qlayer.bias.copy_(linear.bias)
        _update_meta_shape_numel(qlayer)
        return qlayer

    def from_fp32_(
            self,
            weight_fp32: torch.Tensor,
            *,
            scale: Optional[torch.Tensor] = None,
            zero_point: Optional[torch.Tensor] = None,
            bits: Optional[int] = None,
            per_channel: Optional[bool] = None,
    ) -> "QuantizedLinear":
        if not isinstance(weight_fp32, torch.Tensor):
            raise TypeError("from_fp32_: weight_fp32 must be a torch.Tensor")

        with torch.no_grad():
            if tuple(weight_fp32.shape) == tuple(self.weight.shape):
                self.weight.copy_(weight_fp32)
            elif tuple(weight_fp32.t().shape) == tuple(self.weight.shape):
                self.weight.copy_(weight_fp32.t())
            elif weight_fp32.numel() == self.weight.numel():
                self.weight.copy_(weight_fp32.reshape_as(self.weight))
            else:
                raise ValueError(
                    f"from_fp32_: incompatible shapes {tuple(weight_fp32.shape)} vs {tuple(self.weight.shape)}"
                )

        if bits is not None:
            self.bits = _safe_int_or_default(bits, self.bits)
        if per_channel is not None:
            self.per_channel = _safe_bool_or_default(per_channel, self.per_channel)

        _register_optional_buffer(self, "qscale", scale)
        _register_optional_buffer(self, "qzero_point", zero_point)

        if not hasattr(self, "quant_meta"):
            self.quant_meta = {"name": None, "bits": self.bits}
        self.quant_meta["bits"] = self.bits
        _update_meta_shape_numel(self)
        return self


# ========================= Shim: QuantizedEmbedding ========================
class QuantizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        q = _pop_quant_kwargs(kwargs)
        super().__init__(num_embeddings, embedding_dim, **kwargs)

        self.bits: int = _safe_int_or_default(q.get("bits", 8), 8)
        self.per_channel: bool = _safe_bool_or_default(q.get("per_channel", True), True)
        self.group_size: Optional[int] = q.get("group_size", None)

        _register_optional_buffer(self, "qscale", q.get("scale"))
        _register_optional_buffer(self, "qzero_point", q.get("zero_point"))
        _register_optional_buffer(self, "qscales", q.get("scales"))
        _register_optional_buffer(self, "qzero_points", q.get("zero_points"))

        self.quant_meta = {
            "name": None,
            "bits": self.bits,
            "weight_numel": int(self.weight.numel()),
            "shape": tuple(self.weight.shape),
        }

    @classmethod
    def from_embedding(
            cls,
            emb: nn.Embedding,
            bits: int = 8,
            per_channel: bool = True,
            **kwargs,
    ) -> "QuantizedEmbedding":
        qemb = cls(
            emb.num_embeddings, emb.embedding_dim,
            bits=bits, per_channel=per_channel,
            padding_idx=emb.padding_idx, max_norm=emb.max_norm, norm_type=emb.norm_type,
            scale_grad_by_freq=emb.scale_grad_by_freq, sparse=emb.sparse,
            dtype=emb.weight.dtype, device=emb.weight.device, **kwargs
        )
        with torch.no_grad():
            qemb.weight.copy_(emb.weight)
        _update_meta_shape_numel(qemb)
        return qemb

    def from_fp32_(
            self,
            weight_fp32: torch.Tensor,
            *,
            scale: Optional[torch.Tensor] = None,
            zero_point: Optional[torch.Tensor] = None,
            bits: Optional[int] = None,
            per_channel: Optional[bool] = None,
    ) -> "QuantizedEmbedding":
        if not isinstance(weight_fp32, torch.Tensor):
            raise TypeError("from_fp32_: weight_fp32 must be a torch.Tensor")

        with torch.no_grad():
            if tuple(weight_fp32.shape) == tuple(self.weight.shape):
                self.weight.copy_(weight_fp32)
            elif tuple(weight_fp32.t().shape) == tuple(self.weight.shape):
                # Defensive path: rarely needed for embeddings, but harmless.
                self.weight.copy_(weight_fp32.t())
            elif weight_fp32.numel() == self.weight.numel():
                self.weight.copy_(weight_fp32.reshape_as(self.weight))
            else:
                raise ValueError(
                    f"from_fp32_: incompatible shapes {tuple(weight_fp32.shape)} vs {tuple(self.weight.shape)}"
                )

        if bits is not None:
            self.bits = _safe_int_or_default(bits, self.bits)
        if per_channel is not None:
            self.per_channel = _safe_bool_or_default(per_channel, self.per_channel)

        _register_optional_buffer(self, "qscale", scale)
        _register_optional_buffer(self, "qzero_point", zero_point)

        if not hasattr(self, "quant_meta"):
            self.quant_meta = {"name": None, "bits": self.bits}
        self.quant_meta["bits"] = self.bits
        _update_meta_shape_numel(self)
        return self


# =========================== Registry utilities ===========================
def ensure_registry(model: nn.Module) -> None:
    if not hasattr(model, "_variable_bit_registry"):
        setattr(model, "_variable_bit_registry", [])


def register_quantized_layer(model: nn.Module, layer: nn.Module) -> None:
    ensure_registry(model)
    reg: list = getattr(model, "_variable_bit_registry")
    if layer not in reg:
        reg.append(layer)


# ============================ Meta iteration ==============================
def iter_quant_meta_from_model(model: nn.Module) -> Generator[Dict[str, Any], None, None]:
    if hasattr(model, "_variable_bit_registry") and getattr(model, "_variable_bit_registry"):
        for layer in getattr(model, "_variable_bit_registry"):
            meta = getattr(layer, "quant_meta", None)
            if meta is not None:
                if "name" not in meta or meta["name"] in (None, "unknown"):
                    meta["name"] = _infer_layer_name(layer, model)
                if "weight_numel" not in meta and hasattr(layer, "weight") and layer.weight is not None:
                    meta["weight_numel"] = int(layer.weight.numel())
                if "shape" not in meta and hasattr(layer, "weight") and layer.weight is not None:
                    meta["shape"] = tuple(layer.weight.shape)
                yield meta
        return

    for name, module in model.named_modules():
        meta = getattr(module, "quant_meta", None)
        if meta is not None:
            if "name" not in meta:
                meta["name"] = name
            if "weight_numel" not in meta and hasattr(module, "weight") and module.weight is not None:
                meta["weight_numel"] = int(module.weight.numel())
            if "shape" not in meta and hasattr(module, "weight") and module.weight is not None:
                meta["shape"] = tuple(module.weight.shape)
            yield meta


def _infer_layer_name(layer: nn.Module, root: nn.Module) -> str:
    for name, module in root.named_modules():
        if module is layer:
            return name
    return "unknown"


# =========================== Memory accounting ============================
def _bits_to_mib(bits: int) -> float:
    return bits / 8.0 / 1024.0 / 1024.0


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_tuple(x) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(x)
    except Exception:
        return None


def calculate_memory_savings(model: nn.Module) -> Dict[str, Any]:
    total_q_bits = 0
    total_params = 0
    per_layer: List[Dict[str, Any]] = []

    for meta in iter_quant_meta_from_model(model):
        bits = _safe_int(meta.get("bits", 8), 8)
        n = _safe_int(meta.get("weight_numel", 0), 0)
        name = str(meta.get("name", "?"))
        shape = _safe_tuple(meta.get("shape"))

        if n <= 0 or bits <= 0:
            continue

        total_q_bits += bits * n
        total_params += n
        per_layer.append({"name": name, "bits": bits, "params": n, "shape": shape})

    fp32_bits = 32 * total_params
    comp_ratio = float("inf") if total_q_bits == 0 else fp32_bits / float(total_q_bits)

    return {
        "total_params": int(total_params),
        "fp32_MiB": _bits_to_mib(fp32_bits),
        "quant_MiB": _bits_to_mib(total_q_bits),
        "compression_ratio": comp_ratio,
        "layers": per_layer,
    }


# Backward-compatible wrapper
def _iter_quant_meta(model: nn.Module) -> Iterable[Dict[str, Any]]:
    return iter_quant_meta_from_model(model)


# =================== Layer replacement utility (optional) ==================
def _resolve_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[Optional[nn.Module], Optional[str]]:
    try:
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
    except Exception:
        return None, None


def _get_module_by_name(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    name_map = dict(root.named_modules())
    if dotted_name in name_map:
        return name_map[dotted_name]
    parent, attr = _resolve_parent_and_attr(root, dotted_name)
    if parent is not None and attr and hasattr(parent, attr):
        return getattr(parent, attr)
    return None


def _excluded_by_name_or_type(mod_name: str, module: nn.Module) -> bool:
    clsname = module.__class__.__name__
    if "LayerNorm" in clsname or "layernorm" in clsname.lower():
        return True
    if mod_name.endswith("classifier") or "classifier" in mod_name:
        return True
    return False


def replace_linear_with_quantized(
        model: nn.Module,
        name_to_bits: Dict[str, int],
        *,
        per_channel: bool = True,
        also_embeddings: bool = True,
        register: bool = True,
) -> int:
    ensure_registry(model)
    replaced = 0
    name_map = dict(model.named_modules())

    for layer_name, bits in name_to_bits.items():
        candidate_names = [layer_name]
        if layer_name.endswith(".weight") or layer_name.endswith(".bias"):
            candidate_names.append(layer_name.rsplit(".", 1)[0])

        target_module = None
        target_module_name = None
        for cand in candidate_names:
            m = name_map.get(cand) or _get_module_by_name(model, cand)
            if isinstance(m, nn.Module):
                target_module = m
                target_module_name = cand
                break

        if target_module is None or _excluded_by_name_or_type(target_module_name, target_module):
            continue

        parent, attr = _resolve_parent_and_attr(model, target_module_name)
        if parent is None or not attr:
            continue

        if isinstance(target_module, nn.Linear):
            qlayer = QuantizedLinear.from_linear(target_module, bits=int(bits), per_channel=per_channel)
            qlayer.quant_meta["name"] = target_module_name
            setattr(parent, attr, qlayer)
            if register:
                register_quantized_layer(model, qlayer)
            replaced += 1
            continue

        if also_embeddings and isinstance(target_module, nn.Embedding):
            qemb = QuantizedEmbedding.from_embedding(target_module, bits=int(bits), per_channel=False)
            qemb.quant_meta["name"] = target_module_name
            setattr(parent, attr, qemb)
            if register:
                register_quantized_layer(model, qemb)
            replaced += 1
            continue

    return replaced
