"""Export phase for compression pipeline.

Saves the compressed model in various formats (HF-compatible, variable-bit, packed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from ..pipeline.phase_base import CompressionPhase, PhaseContext
from ..quantization import save_quantization_metadata
from ..finalize import finalize_artifacts
from ..utils import make_json_serializable
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


class ExportPhase(CompressionPhase):
    """Export phase: Save compressed model and metadata.
    
    Features:
    - HF-compatible PyTorch format (default)
    - Variable-bit quantized export (runtime savings)
    - Packed format (storage optimization)
    - Comprehensive metadata and statistics
    """
    
    def __init__(self, phase_config: Dict[str, Any]):
        super().__init__(phase_config)
        self.export_variable_bit = phase_config.get('export_variable_bit', True)
        self.export_packed = phase_config.get('export_packed', False)
        self.export_int = phase_config.get('export_int', False)
    
    def get_phase_name(self) -> str:
        return "Export"
    
    def validate(self, context: PhaseContext) -> None:
        """Validate export preconditions."""
        if context.model is None:
            raise ValueError("Model not loaded in context")
        if context.output_path is None:
            raise ValueError("Output path not specified")
    
    def execute(self, context: PhaseContext) -> PhaseContext:
        """Execute export phase."""
        model = context.model
        output_path = context.output_path
        
        compressed_dir = output_path / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Merge quantization and trim metadata
            quant_data = context.metadata.get('quantization', {})
            quant_meta = quant_data.get('parameters', {})
            trim_meta = context.metadata.get('headroom_trimming', {})
            combined_meta = self._merge_meta(quant_meta, trim_meta)
            
            # Build variable-bit registry
            if combined_meta:
                self._build_registry(model, combined_meta, context)
            
            # Variable-bit export (runtime savings)
            variable_bit_stats = None
            if self.export_variable_bit and combined_meta:
                variable_bit_stats = self._export_variable_bit(model, combined_meta, compressed_dir, context)
            
            # Packed export (storage savings)
            packed_stats = None
            if self.export_packed and trim_meta:
                packed_stats = self._export_packed(model, trim_meta, quant_meta, compressed_dir)
            
            # Save metadata
            if quant_meta:
                save_quantization_metadata(quant_meta, compressed_dir)
            
            if trim_meta:
                self._save_trim_metadata(trim_meta, compressed_dir)
            
            # Save compression summary
            summary = self._build_summary(context, variable_bit_stats, packed_stats)
            summary = make_json_serializable(summary)  # Convert Tensors to serializable types
            summary_path = compressed_dir / "compression_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved compression summary to: {summary_path}")
            
            # Finalize artifacts
            finalize_artifacts(
                experiment_dir=output_path,
                keep_tokenizer=True,
                primary_weight="pytorch_model.bin",
                dry_run=False
            )
            
            context.metadata['export'] = {
                'success': True,
                'compressed_dir': str(compressed_dir),
                'variable_bit_stats': variable_bit_stats,
                'packed_stats': packed_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to save compressed model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return context
    
    def _merge_meta(self, quant_meta: Dict, trim_meta: Dict) -> Dict:
        """Merge quantization and trim metadata."""
        combined_meta = {}
        for name in set(list(quant_meta.keys()) + list(trim_meta.keys())):
            if name in quant_meta and name in trim_meta:
                # Deep merge: start with quant_meta, update with trim_meta
                combined_meta[name] = {**quant_meta[name], **trim_meta[name]}
            elif name in quant_meta:
                combined_meta[name] = quant_meta[name]
            else:
                combined_meta[name] = trim_meta[name]
        return combined_meta
    
    def _build_registry(self, model: nn.Module, combined_meta: Dict, context: PhaseContext) -> None:
        """Build variable-bit registry for export."""
        from ..utils.registry import build_variable_bit_registry_from_meta
        
        try:
            compression_cfg = context.config.get('compression', {})
            quantization_cfg = compression_cfg.get('quantization', {})
            
            registered_count = build_variable_bit_registry_from_meta(
                model,
                combined_meta,
                exclude_layernorm_and_classifier=True,
                fallback_bits=quantization_cfg.get('bits_upper', 8),
            )
            logger.info(f"[variable-bit] Registered {registered_count} quantized layers for export")
        except Exception as e:
            logger.warning(f"[variable-bit] Failed to build registry: {e}")
    
    def _export_variable_bit(
        self, 
        model: nn.Module, 
        combined_meta: Dict, 
        output_dir: Path,
        context: PhaseContext
    ) -> Optional[Dict]:
        """Export variable-bit quantized model."""
        from metapac.src.compression.variable_bit_export import integrate_variable_bit_export
        
        logger.info("Exporting with variable bit-width quantization...")
        logger.info("This provides TRUE runtime memory savings (4-16x compression)")
        
        compression_cfg = context.config.get('compression', {})
        baseline_model_config = compression_cfg.get('baseline_model_config')
        
        try:
            stats = integrate_variable_bit_export(
                model,
                combined_meta,
                output_dir,
                export_variable_bit=True,
                use_cuda=torch.cuda.is_available(),
                source_model_path=baseline_model_config
            )
            return stats
        except Exception as e:
            logger.error(f"Variable-bit export failed: {e}")
            return None
    
    def _export_packed(
        self, 
        model: nn.Module, 
        trim_meta: Dict, 
        quant_meta: Dict, 
        output_dir: Path
    ) -> Optional[Dict]:
        """Export packed model (storage optimization)."""
        from metapac.src.compression.bitpacking import save_packed_model
        
        logger.info("Exporting with variable-bit packing...")
        logger.info("Note: This is for disk storage optimization (saves disk space)")
        logger.info("For inference, use pytorch_model.bin (HF-compatible variable-bit model)")
        
        try:
            stats = save_packed_model(
                model.state_dict(),
                trim_meta,
                quant_meta,
                output_dir
            )
            
            logger.info(f"Variable-bit packing complete:")
            logger.info(f"Total parameters: {stats['total_params']:,}")
            logger.info(f"Packed (variable-bit): {stats['packed_params']:,} ({100 * stats['packed_params'] / stats['total_params']:.1f}%)")
            logger.info(f"FP32 (unpacked): {stats['fp32_params']:,} ({100 * stats['fp32_params'] / stats['total_params']:.1f}%)")
            logger.info(f"Original size: {stats['total_original_size'] / (1024 * 1024):.2f} MB")
            logger.info(f"Packed size: {stats['size_mb']:.2f} MB")
            logger.info(f"Compression ratio: {stats['compression_ratio']:.2f}x")
            
            return stats
        except Exception as e:
            logger.error(f"Packed export failed: {e}")
            return None
    
    def _save_trim_metadata(self, trim_meta: Dict, output_dir: Path) -> None:
        """Save headroom trimming metadata."""
        trim_meta_path = output_dir / "headroom_trim_meta.json"
        
        # Convert types for JSON serialization
        converted = self._convert_for_json(trim_meta)
        
        with open(trim_meta_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, indent=2)
        logger.info(f"Saved headroom trimming metadata to: {trim_meta_path}")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._convert_for_json(v) for v in obj)
        if isinstance(obj, torch.Tensor):
            try:
                arr = obj.detach().cpu()
                if arr.numel() == 1:
                    return arr.item()
                return arr.tolist()
            except Exception:
                return None
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        try:
            return float(obj)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return None
    
    def _build_summary(
        self, 
        context: PhaseContext, 
        variable_bit_stats: Optional[Dict],
        packed_stats: Optional[Dict]
    ) -> Dict:
        """Build comprehensive compression summary."""
        compression_cfg = context.config.get('compression', {})
        
        summary = {
            'target_model': compression_cfg.get('target_model'),
            'meta_checkpoint': compression_cfg.get('meta_checkpoint'),
            'pruning': context.metadata.get('pruning', {'enabled': False}),
            'quantization': context.metadata.get('quantization', {'enabled': False, 'num_quantized': 0, 'parameters': {}}),  
            'fine_tuning': context.metadata.get('fine_tuning', {'enabled': False}),
        }
        
        if variable_bit_stats:
            summary['variable_bit_quantization'] = variable_bit_stats
        if packed_stats:
            summary['variable_bit_packing'] = packed_stats
        
        return summary
