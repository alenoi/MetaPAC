"""
Rank-aware quantization with intelligent headroom trimming.

This module implements the final quantization logic for MetaPAC's compression pipeline:
- Rank-aware initial bit assignment
- Intelligent headroom trimming (removes unused codes only)
- Per-tensor and per-channel quantization
- Symmetric and asymmetric modes
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration for quantization."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', True)
        self.mode = config.get('mode', 'rank_aware_trim')
        self.bits_lower = config.get('bits_lower', 4)
        self.bits_upper = config.get('bits_upper', 8)
        self.per_channel = config.get('per_channel', True)
        self.symmetric = config.get('symmetric', True)
        self.clip_percentile = config.get('clip_percentile', 0.0)
        self.mapping = config.get('mapping', {'type': 'linear'})
        self.util_target = config.get('util_target', 0.98)
        self.export_int = config.get('export_int', False)
        self.layer_overrides = config.get('layer_overrides', [])

        # Headroom trimming minimum (can be different from bits_lower)
        # If not set, defaults to bits_lower
        self.headroom_min_bits = config.get('headroom_min_bits', self.bits_lower)

        # Validate
        assert 2 <= self.bits_lower <= self.bits_upper <= 8, \
            f"Invalid bits range: [{self.bits_lower}, {self.bits_upper}]"
        assert 2 <= self.headroom_min_bits <= self.bits_upper, \
            f"Invalid headroom_min_bits: {self.headroom_min_bits}"
        assert 0.0 < self.util_target <= 1.0, \
            f"Invalid util_target: {self.util_target}"


class Quantizer:
    """Quantizer with rank-aware bit assignment and headroom trimming."""

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._quantized_params = set()  # Track already quantized params for idempotency

    def assign_bits_from_rank(self, rank: float, param_name: str = "") -> int:
        """
        Map within-zone rank [0,1] to initial bits in [bits_lower, bits_upper].
        
        Args:
            rank: Normalized rank within zone (0=least important, 1=most important)
            param_name: Parameter name for override lookup
            
        Returns:
            Initial bit width
        """
        # Check for layer-specific override
        for override in self.config.layer_overrides:
            if 'pattern' in override and 'bits' in override:
                import re
                if re.search(override['pattern'], param_name):
                    return override['bits']

        # Apply mapping function
        mapping_type = self.config.mapping.get('type', 'linear')

        if mapping_type == 'linear':
            # Linear interpolation
            bits_float = self.config.bits_lower + rank * (self.config.bits_upper - self.config.bits_lower)
        elif mapping_type == 'sqrt':
            # Square root mapping (more bits for higher ranks)
            bits_float = self.config.bits_lower + np.sqrt(rank) * (self.config.bits_upper - self.config.bits_lower)
        elif mapping_type == 'piecewise':
            # Piecewise linear (example: more granularity in middle range)
            breakpoints = self.config.mapping.get('breakpoints', [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
            # Linear interpolation between breakpoints
            for i in range(len(breakpoints) - 1):
                r1, b1 = breakpoints[i]
                r2, b2 = breakpoints[i + 1]
                if r1 <= rank <= r2:
                    alpha = (rank - r1) / (r2 - r1) if r2 > r1 else 0.0
                    normalized = b1 + alpha * (b2 - b1)
                    bits_float = self.config.bits_lower + normalized * (self.config.bits_upper - self.config.bits_lower)
                    break
        else:
            bits_float = self.config.bits_lower + rank * (self.config.bits_upper - self.config.bits_lower)

        # Round to integer
        bits_init = int(np.round(bits_float))
        bits_init = max(self.config.bits_lower, min(bits_init, self.config.bits_upper))

        return bits_init

    def compute_scale(self, x: torch.Tensor, bits: int,
                      per_channel: bool = False, dim: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute quantization scale using configured method.
        
        Args:
            x: Input tensor
            bits: Bit width
            per_channel: If True, compute per-channel scales
            dim: Channel dimension (typically 0 for weights)
            
        Returns:
            scale: Quantization scale(s)
            meta: Metadata dict with clipping info
        """
        meta = {}

        # Compute qmax for signed symmetric quantization
        qmax = 2 ** (bits - 1) - 1

        # Apply percentile clipping if configured
        if self.config.clip_percentile > 0:
            if per_channel:
                # Per-channel clipping
                abs_x = x.abs()
                # Reshape to (channels, -1)
                shape = list(x.shape)
                abs_x_flat = abs_x.transpose(0, dim).reshape(shape[dim], -1)

                clip_vals = []
                for i in range(abs_x_flat.shape[0]):
                    percentile = torch.quantile(abs_x_flat[i], self.config.clip_percentile)
                    clip_vals.append(percentile)
                clip_val = torch.tensor(clip_vals, device=x.device, dtype=x.dtype)

                meta['clipped'] = True
                meta['clip_percentile'] = self.config.clip_percentile
                meta['clip_values'] = clip_val.tolist()
            else:
                # Per-tensor clipping
                clip_val = torch.quantile(x.abs(), self.config.clip_percentile)
                meta['clipped'] = True
                meta['clip_percentile'] = self.config.clip_percentile
                meta['clip_value'] = clip_val.item()
        else:
            clip_val = None
            meta['clipped'] = False

        # Compute maximum absolute values
        if per_channel:
            # Reduce over all dimensions except channel dim
            dims = list(range(len(x.shape)))
            dims.remove(dim)

            if clip_val is not None:
                x_clipped = x.clamp(-clip_val.view(-1, *([1] * len(dims))),
                                    clip_val.view(-1, *([1] * len(dims))))
                max_val = x_clipped.abs().amax(dim=dims)
            else:
                max_val = x.abs().amax(dim=dims)
        else:
            if clip_val is not None:
                x_clipped = x.clamp(-clip_val, clip_val)
                max_val = x_clipped.abs().max()
            else:
                max_val = x.abs().max()

        # Compute scale: scale = max_val / qmax
        # Add small epsilon to avoid division by zero
        scale = max_val / qmax + 1e-8

        meta['qmax'] = qmax
        meta['max_val'] = max_val.tolist() if isinstance(max_val, torch.Tensor) else max_val.item()

        return scale, meta

    def utilization(self, x: torch.Tensor, bits: int,
                    per_channel: bool = False, dim: int = 0) -> float:
        """
        Compute dynamic range utilization.
        
        Args:
            x: Input tensor
            bits: Bit width
            per_channel: If True, compute per-channel (returns average)
            dim: Channel dimension
            
        Returns:
            Utilization ratio (0-1)
        """
        scale, meta = self.compute_scale(x, bits, per_channel, dim)
        qmax = meta['qmax']
        max_val = meta['max_val']

        if per_channel:
            # Average utilization across channels
            max_val_tensor = torch.tensor(max_val, device=x.device, dtype=x.dtype)
            scale_tensor = scale if isinstance(scale, torch.Tensor) else torch.tensor([scale])
            util = (max_val_tensor / (qmax * scale_tensor)).mean().item()
        else:
            util = max_val / (qmax * scale.item())

        return util

    def trim_headroom_bits(self, x: torch.Tensor, b_init: int,
                           per_channel: bool = False, dim: int = 0,
                           unconstrained: bool = False) -> int:
        """
        Trim bits to remove unused headroom while maintaining util_target.
        
        Args:
            x: Input tensor
            b_init: Initial bit width
            per_channel: If True, use per-channel analysis
            dim: Channel dimension
            unconstrained: If True, use 1 as minimum (deprecated, use headroom_min_bits instead)
            
        Returns:
            Final bit width after trimming
        """
        # Determine minimum bit limit
        # Use headroom_min_bits (can be different from bits_lower for rank-aware assignment)
        if unconstrained:
            min_bits = 1  # Legacy behavior
        else:
            min_bits = self.config.headroom_min_bits

        # Early exit if already at minimum
        if b_init <= min_bits:
            return b_init

        # Binary search for minimal bits achieving util_target
        # Start from min_bits and find the smallest b where util >= target
        b_final = b_init

        for b in range(min_bits, b_init + 1):
            util = self.utilization(x, b, per_channel, dim)
            if util >= self.config.util_target:
                b_final = b
                break

        return b_final

    def trim_headroom_bits_batch(self, x: torch.Tensor, b_init: int,
                                 per_channel: bool = False, dim: int = 0,
                                 unconstrained: bool = False) -> Tuple[int, float, float]:
        """
        Optimized version that returns both final bits and utilizations.
        Reduces redundant computation by computing both util_init and util_final in one pass.
        
        Args:
            x: Input tensor
            b_init: Initial bit width
            per_channel: If True, use per-channel analysis
            dim: Channel dimension
            unconstrained: If True, use 1 as minimum
            
        Returns:
            Tuple of (b_final, util_init, util_final)
        """
        # Determine minimum bit limit
        min_bits = 1 if unconstrained else self.config.headroom_min_bits

        # Early exit if already at minimum
        if b_init <= min_bits:
            util = self.utilization(x, b_init, per_channel, dim)
            return b_init, util, util

        # Compute initial utilization
        util_init = self.utilization(x, b_init, per_channel, dim)

        # If already below target, no trimming possible
        if util_init < self.config.util_target:
            return b_init, util_init, util_init

        # Find minimal bits achieving util_target
        b_final = b_init
        util_final = util_init

        for b in range(min_bits, b_init):  # Don't re-compute b_init
            util = self.utilization(x, b, per_channel, dim)
            if util >= self.config.util_target:
                b_final = b
                util_final = util
                break

        return b_final, util_init, util_final

    def quantize_per_tensor(self, x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize tensor with per-tensor scale.
        
        Args:
            x: Input tensor
            bits: Bit width
            
        Returns:
            q: Quantized tensor (fake-quant or int)
            scale: Quantization scale
            meta: Metadata
        """
        scale, meta = self.compute_scale(x, bits, per_channel=False)
        qmax = meta['qmax']

        # Quantize
        if self.config.symmetric:
            # Symmetric: q = round(x / scale)
            q = torch.round(x / scale).clamp(-qmax, qmax)
            zero_point = torch.tensor(0.0, device=x.device, dtype=scale.dtype)
        else:
            # Asymmetric: q = round(x / scale) + zero_point
            qmin = 0
            qmax_asym = 2 ** bits - 1
            zero_point = qmin - torch.round(x.min() / scale)
            q = torch.round(x / scale + zero_point).clamp(qmin, qmax_asym)
            meta['zero_point'] = zero_point.item()

        # Store symmetric flag in metadata for INT8 export
        meta['symmetric'] = self.config.symmetric
        meta['zero_point'] = int(zero_point.item()) if zero_point is not None else 0

        # ALWAYS store the quantized int values for lossless packing
        # This allows packed export without re-quantization errors
        q_int = q.clone().detach().cpu()  # Store as CPU tensor to save memory
        meta['_q_int'] = q_int

        if self.config.export_int:
            # Return int8 tensors directly (for export_int mode)
            return q.to(torch.int8), scale, meta

        # Default fake-quant FP32 path (for training-time simulation)
        if self.config.symmetric:
            q_deq = q * scale
        else:
            q_deq = (q - zero_point) * scale

        return q_deq, scale, meta

    def quantize_per_channel(self, x: torch.Tensor, bits: int, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize tensor with per-channel scales.
        
        Args:
            x: Input tensor
            bits: Bit width
            dim: Channel dimension (0 for out_features/out_channels)
            
        Returns:
            q: Quantized tensor
            scale: Per-channel scales
            meta: Metadata
        """
        scale, meta = self.compute_scale(x, bits, per_channel=True, dim=dim)
        qmax = meta['qmax']

        # Reshape scale for broadcasting
        shape = [1] * len(x.shape)
        shape[dim] = x.shape[dim]
        scale_view = scale.view(shape)

        # Quantize
        if self.config.symmetric:
            q = torch.round(x / scale_view).clamp(-qmax, qmax)
            zero_point = torch.zeros(scale.size(), device=x.device, dtype=scale.dtype)
            zero_point_view = zero_point.view(shape)
        else:
            # Per-channel asymmetric
            qmin = 0
            qmax_asym = 2 ** bits - 1
            x_min = x.transpose(0, dim).reshape(x.shape[dim], -1).min(dim=1)[0]
            zero_point = qmin - torch.round(x_min / scale)
            zero_point_view = zero_point.view(shape)
            q = torch.round(x / scale_view + zero_point_view).clamp(qmin, qmax_asym)
            meta['zero_point'] = zero_point.tolist()

        # Store symmetric flag in metadata for INT8 export
        meta['symmetric'] = self.config.symmetric
        meta['zero_point'] = zero_point.tolist() if zero_point.numel() > 1 else int(zero_point.item())

        # ALWAYS store the quantized int values for lossless packing
        # This allows packed export without re-quantization errors
        q_int = q.clone().detach().cpu()  # Store as CPU tensor to save memory
        meta['_q_int'] = q_int

        if self.config.export_int:
            # Return int8 tensors directly (for export_int mode)
            return q.to(torch.int8), scale, meta

        # Default fake-quant FP32 path
        if self.config.symmetric:
            q_deq = q * scale_view
        else:
            q_deq = (q - zero_point_view) * scale_view

        return q_deq, scale, meta

    def apply_quantization(
            self,
            model: nn.Module,
            plan: Dict[str, str],
            importance_rankings: Dict[str, float],
            target_bits_map: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply quantization to model parameters in the 'quantize' zone.
        
        Args:
            model: PyTorch model
            plan: Parameter -> zone mapping ('keep', 'quantize', 'prune')
            importance_rankings: Parameter -> importance score [0,1]
            target_bits_map: Parameter -> target bits (zone-specific), None means use rank-aware
            
        Returns:
            Metadata dict with quantization info per parameter
        """
        if target_bits_map is None:
            target_bits_map = {}

        quant_meta = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                # Check for force_quantize override
                force_quantize = False
                for override in self.config.layer_overrides:
                    if 'pattern' in override and override.get('force_quantize', False):
                        import re
                        if re.search(override['pattern'], name):
                            force_quantize = True
                            break

                # Skip if not in quantize zone (unless forced)
                if not force_quantize and plan.get(name) != 'quantize':
                    continue

                # Check idempotency
                param_id = id(param)
                if param_id in self._quantized_params:
                    logger.warning(f"Parameter {name} already quantized, skipping")
                    continue

                # Get rank (normalized importance within quantize zone)
                rank = importance_rankings.get(name, 0.5)

                # Check for zone-specific target bits
                override_bits = target_bits_map.get(name)

                if override_bits is None:
                    b_init = self.assign_bits_from_rank(rank, name)
                elif isinstance(override_bits, float) and np.isnan(override_bits):
                    b_init = self.assign_bits_from_rank(rank, name)
                else:
                    b_init = int(override_bits)

                # Compute initial utilization
                per_channel = self.config.per_channel and (param.dim() >= 2)
                dim = 0  # Quantize over out_features/out_channels

                util_init = self.utilization(param.data, b_init, per_channel, dim)

                # Trim headroom
                b_final = self.trim_headroom_bits(param.data, b_init, per_channel, dim)

                # Compute final utilization
                util_final = self.utilization(param.data, b_final, per_channel, dim)

                # Apply quantization
                if per_channel:
                    q, scale, meta = self.quantize_per_channel(param.data, b_final, dim)
                else:
                    q, scale, meta = self.quantize_per_tensor(param.data, b_final)

                # If the quantizer returned integer tensors (export_int path), dequantize
                # back to floating point before assigning to parameter.data. Assigning
                # a non-floating dtype to a tensor that requires gradients raises:
                # "data set to a tensor that requires gradients must be floating point or complex dtype".
                try:
                    is_floating = q.dtype.is_floating_point
                except Exception:
                    # Fallback: assume floating
                    is_floating = True

                if not is_floating:
                    # Dequantize using available scale / zero_point metadata
                    if isinstance(scale, torch.Tensor):
                        # Per-channel scale: reshape for broadcasting when possible.
                        shape = [1] * len(param.shape)
                        shape[dim] = param.shape[dim]

                        # If scale is scalar (numel == 1), broadcast the scalar to the expected shape.
                        try:
                            scale_numel = scale.numel()
                        except Exception:
                            scale_numel = 1

                        if scale_numel == 1:
                            scalar_scale = float(scale.item())
                            scale_view = torch.full(shape, scalar_scale, dtype=param.dtype, device=param.device)
                        else:
                            # If scale tensor length matches channel dim, reshape; otherwise fall back to broadcasting scalar.
                            if scale_numel == param.shape[dim]:
                                scale_view = scale.view(shape).to(dtype=param.dtype, device=param.device)
                            else:
                                # Unexpected shape: broadcast mean of scale or first element
                                scalar_scale = float(scale.mean().item()) if scale_numel > 0 else float(scale.item())
                                scale_view = torch.full(shape, scalar_scale, dtype=param.dtype, device=param.device)

                        if meta.get('symmetric', True):
                            q = q.to(scale_view.dtype) * scale_view
                        else:
                            zp = meta.get('zero_point', 0)
                            if isinstance(zp, (list, tuple)):
                                zp_tensor = torch.tensor(zp, device=param.device, dtype=scale.dtype)
                                # zp may also need broadcasting
                                try:
                                    zp_numel = zp_tensor.numel()
                                except Exception:
                                    zp_numel = 1

                                if zp_numel == param.shape[dim]:
                                    zp_view = zp_tensor.view(shape).to(dtype=param.dtype, device=param.device)
                                else:
                                    zp_view = torch.full(shape, float(zp_tensor.mean().item()), dtype=param.dtype,
                                                         device=param.device)

                                q = (q.to(scale_view.dtype) - zp_view) * scale_view
                            else:
                                q = (q.to(scale_view.dtype) - float(zp)) * scale_view
                    else:
                        # Per-tensor scalar scale
                        scale_val = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.item())
                        if meta.get('symmetric', True):
                            q = q.to(param.dtype) * scale_val
                        else:
                            zp = meta.get('zero_point', 0)
                            q = (q.to(param.dtype) - float(zp)) * scale_val

                # Update parameter in-place (ensure float dtype)
                param.data = q.to(param.dtype)

                # Store metadata (including _q_int from meta if available)
                quant_meta[name] = {
                    'rank': rank,
                    'bits_init': b_init,
                    'bits_final': b_final,
                    'util_init': util_init,
                    'util_final': util_final,
                    'per_channel': per_channel,
                    'scale': scale.tolist() if isinstance(scale, torch.Tensor) else scale.item(),
                    'qmax': meta['qmax'],
                    'shape': list(param.shape),
                    **meta  # This includes '_q_int' if present
                }

                # Mark as quantized
                self._quantized_params.add(param_id)

                # Log
                logger.info(
                    f"Quantized {name}: rank={rank:.3f}, {b_init}b→{b_final}b, "
                    f"util {util_init:.3f}→{util_final:.3f}, "
                    f"per_channel={per_channel}"
                )

        return quant_meta

    def apply_headroom_trimming_all_zones(
            self,
            model: nn.Module,
            plan: Dict[str, str],
            target_bits_map: Optional[Dict[str, Optional[float]]] = None,
            importance_rankings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Apply headroom trimming to ALL parameters across all zones (keep, quantize, prune).
        NOW WITH RANK-AWARE BIT ASSIGNMENT for all zones.
        
        This runs AFTER pruning to optimize bit allocation for all parameters.
        Parameters get rank-aware initial bits, then headroom trimming optimizes further.
        
        Args:
            model: PyTorch model
            plan: Parameter -> zone mapping ('keep', 'quantize', 'prune')
            target_bits_map: Parameter -> target bits (optional, overrides rank-aware assignment)
            importance_rankings: Parameter -> importance score (0-1, used for rank-aware assignment)
            
        Returns:
            Metadata dict with headroom trimming info per parameter
        """
        if target_bits_map is None:
            target_bits_map = {}

        if importance_rankings is None:
            importance_rankings = {}

        trim_meta = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                zone = plan.get(name, 'keep')

                # Determine initial bit width based on zone and importance
                override_bits = target_bits_map.get(name)

                if override_bits is not None:
                    # Explicit target overrides everything
                    b_init = int(override_bits)
                else:
                    # Rank-aware assignment for ALL zones
                    rank = importance_rankings.get(name, 0.5)  # Default to middle rank
                    # Clamp rank to [0, 1] to prevent invalid bit assignments
                    rank = max(0.0, min(1.0, rank))

                    if zone == 'prune':
                        # Pruned params have many zeros, use lower range
                        # Map rank to [headroom_min_bits, bits_lower]
                        # FIXED: Removed +1 to prevent exceeding bits_lower
                        b_init = int(self.config.headroom_min_bits +
                                     rank * (self.config.bits_lower - self.config.headroom_min_bits))
                        # Ensure within valid range [headroom_min_bits, bits_lower]
                        b_init = max(self.config.headroom_min_bits, min(self.config.bits_lower, b_init))
                    else:
                        # Keep and quantize zones: use full rank-aware range
                        # Map rank to [bits_lower, bits_upper]
                        b_init = self.assign_bits_from_rank(rank, name)

                # Validate bit width constraints for CUDA compatibility
                if b_init < 2 or b_init > 8:
                    logger.warning(
                        f"Invalid b_init={b_init} for {name} (zone={zone}, rank={rank:.3f}). "
                        f"Clamping to [2, 8] range."
                    )
                    b_init = max(2, min(8, b_init))

                # Compute per-channel setting
                per_channel = self.config.per_channel and (param.dim() >= 2)
                dim = 0  # Quantize over out_features/out_channels

                # OPTIMIZED: Use batch version to compute both utils in one pass
                b_final, util_init, util_final = self.trim_headroom_bits_batch(
                    param.data, b_init, per_channel, dim, unconstrained=False
                )

                # Store metadata
                trim_meta[name] = {
                    'zone': zone,
                    'bits_init': b_init,
                    'bits_final': b_final,
                    'util_init': util_init,
                    'util_final': util_final,
                    'bits_saved': b_init - b_final,
                    'per_channel': per_channel,
                    'shape': list(param.shape),
                    'importance_rank': importance_rankings.get(name, 0.5)
                }

                # Log significant savings
                if b_init - b_final >= 2:
                    logger.info(
                        f"Headroom trimming {name} ({zone}, rank={importance_rankings.get(name, 0.5):.2f}): "
                        f"{b_init}b→{b_final}b (util {util_init:.3f}→{util_final:.3f})"
                    )

        return trim_meta


def save_quantization_metadata(meta: Dict[str, Any], output_dir: Path):
    """Save quantization metadata to JSON."""
    import numpy as np
    import torch

    def convert_to_json_serializable(obj):
        """Convert numpy/torch types to Python native types for JSON serialization."""
        # Handle PyTorch tensors
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        # Handle numpy types
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Recursively handle collections
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        # Handle basic types
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # Skip non-serializable types
        else:
            logger.warning(f"Skipping non-serializable type: {type(obj)}")
            return None

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / 'quant_meta.json'

    # Filter out non-JSON-serializable items like _q_int tensors (only needed in-memory for packing)
    clean_meta = {}
    for name, param_meta in meta.items():
        if isinstance(param_meta, dict):
            # Filter out _q_int and any other non-serializable keys
            filtered = {k: v for k, v in param_meta.items() if k != '_q_int'}
            clean_meta[name] = filtered
        else:
            clean_meta[name] = param_meta

    # Convert all types to JSON-serializable format
    meta_converted = convert_to_json_serializable(clean_meta)

    # Save with error handling
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta_converted, f, indent=2)
        logger.info(f"Saved quantization metadata to {meta_path}")
    except Exception as e:
        logger.error(f"Failed to save quantization metadata: {e}")
        # Save a minimal version for debugging
        debug_path = output_dir / 'quant_meta_debug.txt'
        with open(debug_path, 'w') as f:
            f.write(str(meta_converted))
        raise


def load_quantization_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load quantization metadata from JSON."""
    meta_path = checkpoint_dir / 'quant_meta.json'

    if not meta_path.exists():
        raise FileNotFoundError(f"Quantization metadata not found: {meta_path}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    logger.info(f"Loaded quantization metadata from {meta_path}")
    return meta
