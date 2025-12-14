"""Variable-bit packing for adaptive quantization.

This module implements true variable-bit storage where each parameter
uses exactly the number of bits determined by headroom trimming.

Example:
    2-bit params: 4 values packed into 1 byte
    3-bit params: 8 values packed into 3 bytes
    4-bit params: 2 values packed into 1 byte
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch


def pack_bits(values: np.ndarray, bits: int) -> bytes:
    """Pack values into compact bit representation.
    
    Args:
        values: Integer values to pack (must fit in `bits` bits)
        bits: Number of bits per value
        
    Returns:
        Packed bytes
        
    Example:
        >>> pack_bits(np.array([0, 1, 2, 3], dtype=np.uint8), bits=2)
        b'\\xe4'  # 0b11100100 = [00, 01, 10, 11]
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"Bits must be in range [1, 8], got {bits}")

    values = values.flatten()
    n = len(values)

    # Calculate total bits needed
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8  # Round up to nearest byte

    # Create output buffer
    packed = bytearray(total_bytes)

    # Pack values bit by bit
    bit_pos = 0
    for val in values:
        # Ensure value fits in bits
        if val < 0 or val >= (1 << bits):
            raise ValueError(f"Value {val} doesn't fit in {bits} bits")

        # Pack this value
        for i in range(bits):
            if val & (1 << (bits - 1 - i)):  # Check bit from MSB to LSB
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)  # MSB first
                packed[byte_idx] |= (1 << bit_idx)
            bit_pos += 1

    return bytes(packed)


def unpack_bits(packed: bytes, bits: int, count: int) -> np.ndarray:
    """Unpack values from compact bit representation.
    
    Args:
        packed: Packed bytes
        bits: Number of bits per value
        count: Number of values to unpack
        
    Returns:
        Unpacked integer array
        
    Example:
        >>> unpack_bits(b'\\xe4', bits=2, count=4)
        array([0, 1, 2, 3], dtype=uint8)
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"Bits must be in range [1, 8], got {bits}")

    # Create output buffer
    values = np.zeros(count, dtype=np.uint8)

    # Unpack values bit by bit
    bit_pos = 0
    for i in range(count):
        val = 0
        for j in range(bits):
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)  # MSB first

            if byte_idx < len(packed):
                if packed[byte_idx] & (1 << bit_idx):
                    val |= (1 << (bits - 1 - j))
            bit_pos += 1

        values[i] = val

    return values


def quantize_and_pack(
        tensor: torch.Tensor,
        bits: int,
        scale: torch.Tensor,
        symmetric: bool = True,
        zero_point: torch.Tensor = None
) -> Tuple[bytes, Dict[str, Any]]:
    """Quantize tensor and pack to variable-bit representation.
    
    Args:
        tensor: Input FP32 tensor
        bits: Target bit width (1-8)
        scale: Quantization scale(s)
        symmetric: Use symmetric quantization
        zero_point: Zero point for asymmetric quantization
        
    Returns:
        Tuple of:
        - packed: Packed bytes
        - meta: Metadata (shape, bits, scale, etc.)
    """
    # Compute qmax
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
    else:
        qmax = 2 ** bits - 1
        qmin = 0

    # Quantize
    if symmetric:
        q = torch.round(tensor / scale).clamp(qmin, qmax)
        # Shift to unsigned range [0, 2^bits-1]
        # CRITICAL: offset must be 2^(bits-1), NOT qmax (which is 2^(bits-1)-1)
        offset = 2 ** (bits - 1)
        q_unsigned = (q + offset).to(torch.int64)
    else:
        q = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        q_unsigned = q.to(torch.int64)

    # Convert to numpy and pack
    q_np = q_unsigned.cpu().numpy().astype(np.uint8)
    packed = pack_bits(q_np, bits)

    # Create metadata
    meta = {
        'shape': list(tensor.shape),
        'bits': bits,
        'symmetric': symmetric,
        'scale': scale.tolist() if isinstance(scale, torch.Tensor) else scale,
        'packed_size': len(packed),
        'original_size': tensor.numel() * 4,  # FP32
        'compression_ratio': (tensor.numel() * 4) / len(packed)
    }

    if not symmetric and zero_point is not None:
        meta['zero_point'] = zero_point.tolist() if isinstance(zero_point, torch.Tensor) else zero_point

    return packed, meta


def unpack_and_dequantize(
        packed: bytes,
        meta: Dict[str, Any],
        device: str = 'cpu'
) -> torch.Tensor:
    """Unpack and dequantize tensor from variable-bit representation.
    
    Args:
        packed: Packed bytes
        meta: Metadata from quantize_and_pack
        device: Target device
        
    Returns:
        Dequantized FP32 tensor
    """
    shape = meta['shape']
    bits = meta['bits']
    symmetric = meta['symmetric']
    scale = torch.tensor(meta['scale'], device=device, dtype=torch.float32)

    # Unpack
    count = int(np.prod(shape))
    q_np = unpack_bits(packed, bits, count)
    q_unsigned = torch.from_numpy(q_np).to(device).to(torch.float32)

    # Reshape first
    q_unsigned = q_unsigned.reshape(shape)

    # Dequantize (scale might be per-channel)
    if symmetric:
        # Shift back to signed range: subtract the same offset used during packing
        offset = 2 ** (bits - 1)
        q = q_unsigned - offset  # Shift back to signed range: e.g., 0..31 -> -16..+15

        # Handle per-channel scale
        if scale.dim() > 0 and len(scale) > 1:
            # Per-channel: reshape scale for broadcasting
            scale_shape = [1] * len(shape)
            scale_shape[0] = shape[0]  # Assume channel is dim 0
            scale = scale.view(scale_shape)

        tensor = q * scale
    else:
        zero_point = torch.tensor(meta['zero_point'], device=device, dtype=torch.float32)

        # Handle per-channel scale and zero_point
        if scale.dim() > 0 and len(scale) > 1:
            # Per-channel: reshape scale for broadcasting
            scale_shape = [1] * len(shape)
            scale_shape[0] = shape[0]  # Assume channel is dim 0
            scale = scale.view(scale_shape)
        
        if zero_point.dim() > 0 and len(zero_point) > 1:
            zero_point_shape = [1] * len(shape)
            zero_point_shape[0] = shape[0]
            zero_point = zero_point.view(zero_point_shape)

        tensor = (q_unsigned - zero_point) * scale

    return tensor


def save_packed_model(
        state_dict: Dict[str, torch.Tensor],
        trim_meta: Dict[str, Dict[str, Any]],
        quant_meta: Dict[str, Dict[str, Any]],
        output_dir: Path
) -> Dict[str, Any]:
    """Save model with variable-bit packing.
    
    Args:
        state_dict: Model state dict (fake-quant FP32)
        trim_meta: Headroom trimming metadata (contains final bits)
        quant_meta: Quantization metadata (contains scales)
        output_dir: Output directory
        
    Returns:
        Statistics about packing
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    packed_params = {}
    metadata = {}
    stats = {
        'total_params': 0,
        'packed_params': 0,
        'fp32_params': 0,
        'total_original_size': 0,
        'total_packed_size': 0,
        'total_fp32_size': 0
    }

    for name, param in state_dict.items():
        stats['total_params'] += param.numel()
        original_size = param.numel() * 4  # FP32 bytes
        stats['total_original_size'] += original_size

        # Check if this param was processed by headroom trimming
        # ALL zones (keep, quantize, prune) have bits_final after headroom trimming
        if name in trim_meta:
            bits = trim_meta[name]['bits_final']
            
            # Strategy:
            # 1. If param is in quant_meta AND has stored int values, use those directly (LOSSLESS!)
            # 2. If param is in quant_meta but no int values, use scale to re-quantize (lossy)
            # 3. Otherwise, compute fresh scale from current FP32 values (keep/prune zones)
            
            if name in quant_meta and '_q_int' in quant_meta[name]:
                # BEST CASE: Use stored quantized int values (NO re-quantization error!)
                qmeta = quant_meta[name]
                q_int = qmeta['_q_int']  # Already on CPU
                scale = qmeta['scale']
                symmetric = qmeta.get('symmetric', True)
                zero_point = qmeta.get('zero_point', None)
                
                # Pack the original int values directly
                if isinstance(q_int, torch.Tensor):
                    # Symmetric: signed int (-qmax .. qmax), need to shift to unsigned
                    # Asymmetric: already unsigned (0 .. qmax_asym)
                    if symmetric:
                        # Shift signed values to unsigned: add 2^(bits-1)
                        offset = 2 ** (bits - 1)
                        q_np = (q_int + offset).numpy().astype(np.uint8)
                    else:
                        # Already unsigned, no shift needed
                        q_np = q_int.numpy().astype(np.uint8)
                else:
                    # Fallback: shouldn't happen
                    raise ValueError(f"Expected tensor for _q_int in {name}")
                
                packed = pack_bits(q_np.flatten(), bits)
                
                # Create metadata
                pack_meta = {
                    'shape': list(param.shape),
                    'bits': bits,
                    'symmetric': symmetric,
                    'scale': scale if not isinstance(scale, torch.Tensor) else scale.tolist(),
                    'packed_size': len(packed),
                    'original_size': param.numel() * 4,
                    'compression_ratio': (param.numel() * 4) / len(packed)
                }
                if zero_point is not None:
                    pack_meta['zero_point'] = zero_point
                
            elif name in quant_meta:
                # FALLBACK: Re-quantize using FRESH scale from current weights
                # (original scale may be stale after fine-tuning)
                qmeta = quant_meta[name]
                symmetric = qmeta.get('symmetric', True)
                per_channel = qmeta.get('per_channel', False)
                
                # Compute FRESH scale from current parameter values
                if symmetric:
                    qmax = 2 ** (bits - 1) - 1
                    if per_channel and len(param.shape) >= 2:
                        # Per-channel: scale per output channel (dim 0)
                        scale = param.abs().amax(dim=tuple(range(1, len(param.shape)))) / qmax
                    else:
                        # Per-tensor
                        scale = param.abs().max() / qmax
                    zero_point = None
                else:
                    qmax = 2 ** bits - 1
                    if per_channel and len(param.shape) >= 2:
                        pmin = param.amin(dim=tuple(range(1, len(param.shape))))
                        pmax = param.amax(dim=tuple(range(1, len(param.shape))))
                        scale = (pmax - pmin) / qmax
                        zero_point = -pmin / scale
                    else:
                        pmin = param.min()
                        pmax = param.max()
                        scale = (pmax - pmin) / qmax
                        zero_point = -pmin / scale
                
                # Re-quantize with FRESH scale
                packed, pack_meta = quantize_and_pack(param, bits, scale, symmetric, zero_point)
            else:
                # Compute fresh scale for keep/prune zones
                # Use symmetric quantization for simplicity
                qmax = 2 ** (bits - 1) - 1
                scale = param.abs().max() / qmax
                symmetric = True
                zero_point = None
                
                # Quantize and pack
                packed, pack_meta = quantize_and_pack(param, bits, scale, symmetric, zero_point)

            packed_params[name] = packed
            metadata[name] = pack_meta

            stats['packed_params'] += param.numel()
            stats['total_packed_size'] += len(packed)
        else:
            # Keep as FP32
            packed_params[name] = param.cpu().numpy().tobytes()
            metadata[name] = {
                'shape': list(param.shape),
                'bits': 32,
                'dtype': 'fp32',
                'packed_size': original_size,
                'original_size': original_size,
                'compression_ratio': 1.0
            }

            stats['fp32_params'] += param.numel()
            stats['total_fp32_size'] += original_size

    # Save packed data
    packed_path = output_dir / 'model_packed.bin'
    with open(packed_path, 'wb') as f:
        # Write header
        import struct
        f.write(struct.pack('I', len(packed_params)))  # Number of parameters

        # Write each parameter
        for name, packed in packed_params.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))  # Name length
            f.write(name_bytes)  # Name
            f.write(struct.pack('I', len(packed)))  # Data length
            f.write(packed)  # Data

    # Save metadata
    meta_path = output_dir / 'packing_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Calculate final stats
    stats['compression_ratio'] = stats['total_original_size'] / (stats['total_packed_size'] + stats['total_fp32_size'])
    stats['size_mb'] = (stats['total_packed_size'] + stats['total_fp32_size']) / (1024 * 1024)

    return stats


def load_packed_model(
        model_dir: Path,
        device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Load model from variable-bit packed format.
    
    Args:
        model_dir: Directory containing packed model
        device: Target device
        
    Returns:
        State dict with FP32 tensors
    """
    # Load metadata
    meta_path = model_dir / 'packing_metadata.json'
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Load packed data
    packed_path = model_dir / 'model_packed.bin'
    state_dict = {}

    with open(packed_path, 'rb') as f:
        import struct

        # Read header
        num_params = struct.unpack('I', f.read(4))[0]

        # Read each parameter
        for _ in range(num_params):
            # Read name
            name_len = struct.unpack('I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            # Read data
            data_len = struct.unpack('I', f.read(4))[0]
            packed = f.read(data_len)

            # Unpack
            meta = metadata[name]
            if meta.get('dtype') == 'fp32':
                # FP32 parameter
                tensor = torch.from_numpy(np.frombuffer(packed, dtype=np.float32))
                tensor = tensor.reshape(meta['shape']).to(device)
            else:
                # Variable-bit packed parameter
                tensor = unpack_and_dequantize(packed, meta, device)

            state_dict[name] = tensor

    return state_dict
