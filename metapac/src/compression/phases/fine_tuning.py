"""Fine-tuning phase for compression pipeline.

Post-compression recovery training to restore accuracy after pruning/quantization.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from ..pipeline.phase_base import CompressionPhase, PhaseContext
from ...utils.logging_utils import get_logger, log_section, log_metric

logger = get_logger(__name__)


class FineTuningPhase(CompressionPhase):
    """Fine-tuning phase: Recovery training after compression.
    
    Features:
    - Knowledge distillation from uncompressed teacher
    - Standard supervised fine-tuning
    - Automatic re-quantization after weight updates
    """
    
    def __init__(self, phase_config: Dict[str, Any]):
        super().__init__(phase_config)
    
    def get_phase_name(self) -> str:
        return "Fine-Tuning"
    
    def validate(self, context: PhaseContext) -> None:
        """Validate fine-tuning preconditions."""
        if context.model is None:
            raise ValueError("Model not loaded in context")
        
        # Check for physical pruning conflict
        pruning_config = context.config.get('compression', {}).get('pruning', {})
        if pruning_config.get('enabled', False) and pruning_config.get('physical', True):
            logger.warning("Physical pruning active - fine-tuning may have issues")
            logger.info("Recommended: Use soft pruning (physical: false) with fine-tuning")
    
    def execute(self, context: PhaseContext) -> PhaseContext:
        """Execute fine-tuning phase."""
        if not self.enabled:
            logger.info("Fine-tuning disabled, skipping")
            context.metadata['fine_tuning'] = {'enabled': False}
            return context
        
        # Check for quantization requirement
        quant_meta = context.metadata.get('quantization', {})
        quant_enabled = quant_meta.get('enabled', False)
        num_quantized = quant_meta.get('num_quantized', 0)
        
        if not quant_enabled or num_quantized == 0:
            logger.info(f"Quantization not applied (enabled={quant_enabled}, num_quantized={num_quantized}), skipping fine-tuning")
            context.metadata['fine_tuning'] = {'enabled': False, 'reason': 'quantization_disabled'}
            return context
        
        model = context.model
        output_path = context.output_path
        compression_cfg = context.config.get('compression', {})
        baseline_model_config = compression_cfg.get('baseline_model_config')
        
        try:
            from metapac.src.compression.fine_tune import run_fine_tuning
            
            # Save pre-fine-tuning quantized model
            quantized_dir = output_path / "quantized_before_ft"
            quantized_dir.mkdir(parents=True, exist_ok=True)
            
            quantized_model_path = quantized_dir / "pytorch_model.bin"
            torch.save(model.state_dict(), quantized_model_path)
            logger.info(f"Saved quantized model to: {quantized_model_path}")
            
            # Copy config and tokenizer files from baseline
            self._copy_model_artifacts(baseline_model_config, quantized_dir)
            
            # Save compression summary
            compression_summary = {
                'target_model': compression_cfg.get('target_model'),
                'pruning_applied': compression_cfg.get('pruning', {}).get('enabled', False),
                'quantization_applied': True
            }
            with open(quantized_dir / "compression_summary.json", 'w') as f:
                json.dump(compression_summary, f, indent=2)
            
            # Prepare fine-tuning config
            ft_config = self._build_fine_tuning_config(quantized_dir, compression_cfg)
            
            logger.info(f"Model checkpoint: {quantized_dir}")
            logger.info(f"Output directory: {ft_config['output_dir']}")
            logger.info(f"Epochs: {ft_config['training']['num_epochs']}")
            logger.info(f"Learning rate: {ft_config['training']['learning_rate']}")
            
            # Run fine-tuning
            result = run_fine_tuning(ft_config)
            
            if result == 0:
                self._load_and_requantize(model, ft_config, context)
            else:
                logger.warning("Fine-tuning failed, continuing with quantized weights")
                context.metadata['fine_tuning'] = {'enabled': True, 'success': False}
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            context.metadata['fine_tuning'] = {'enabled': True, 'success': False, 'error': str(e)}
        
        return context
    
    def _copy_model_artifacts(self, baseline_path: str, target_dir: Path) -> None:
        """Copy config and tokenizer files from baseline model."""
        baseline_dir = Path(baseline_path)
        
        # Copy config
        config_path = baseline_dir / "config.json"
        if config_path.exists():
            shutil.copy(config_path, target_dir / "config.json")
            logger.info("Copied config.json from baseline")
        
        # Copy tokenizer files
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]
        
        copied_count = 0
        for filename in tokenizer_files:
            src = baseline_dir / filename
            if src.exists():
                shutil.copy(src, target_dir / filename)
                copied_count += 1
        
        if copied_count > 0:
            logger.info(f"Copied {copied_count} tokenizer files from baseline")
    
    def _build_fine_tuning_config(self, model_path: Path, compression_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build fine-tuning configuration."""
        ft_cfg = self.config
        
        return {
            'model_checkpoint': str(model_path),
            'output_dir': ft_cfg.get('output_dir', f"{compression_cfg.get('output_dir', 'output')}/finetuned"),
            'data': ft_cfg.get('data', {
                'dataset': 'glue',
                'dataset_config': 'sst2',
                'max_length': 128,
                'batch_size': 8
            }),
            'training': ft_cfg.get('training', {
                'num_epochs': 3,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'gradient_clip': 1.0,
                'device': 'auto',
                'num_workers': 0
            }),
            'distillation': ft_cfg.get('distillation', {
                'enabled': False,
                'teacher_model': compression_cfg.get('baseline_model_config'),
                'temperature': 4.0,
                'alpha': 0.5
            })
        }
    
    def _load_and_requantize(self, model: nn.Module, ft_config: Dict[str, Any], context: PhaseContext) -> None:
        """Load fine-tuned weights and re-quantize."""
        output_dir = Path(ft_config['output_dir'])
        
        # Find fine-tuned model file
        finetuned_model_path = output_dir / "pytorch_model.bin"
        if not finetuned_model_path.exists():
            finetuned_model_path = output_dir / "model.safetensors"
        
        if not finetuned_model_path.exists():
            logger.warning(f"Fine-tuned model not found at {output_dir}")
            return
        
        logger.info(f"Loading fine-tuned weights from: {finetuned_model_path}")
        
        # Load weights
        if finetuned_model_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            finetuned_state = load_file(finetuned_model_path, device='cpu')
        else:
            finetuned_state = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(finetuned_state, strict=False)
        logger.info("✓ Fine-tuned weights loaded successfully")
        
        # Re-quantize with same bit allocations
        if 'quantization' in context.metadata and 'quantizer' in dir(context):
            self._requantize_model(model, context)
        
        # Load metrics
        results_path = output_dir / "fine_tune_results.json"
        if results_path.exists():
            with open(results_path) as f:
                ft_results = json.load(f)
            logger.info(f"Fine-tuning metrics:")
            logger.info(f"Best accuracy: {ft_results.get('best_val_accuracy', 'N/A')}")
            logger.info(f"Final accuracy: {ft_results.get('final_val_accuracy', 'N/A')}")
            context.metadata['fine_tuning'] = {
                'enabled': True, 
                'success': True,
                'metrics': ft_results
            }
        
        # Log post-fine-tuning statistics
        log_section(logger, "After Fine-Tuning")
        after_ft_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        log_metric(logger, "Size", f"{after_ft_size_mb:.2f}", "MB")
        non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
        log_metric(logger, "Non-zero parameters", non_zero_params)
    
    def _requantize_model(self, model: nn.Module, context: PhaseContext) -> None:
        """Re-quantize model after fine-tuning to update quantization metadata."""
        logger.info("Re-quantizing model after fine-tuning...")
        
        quant_meta = context.metadata.get('quantization', {})
        quantizer = None  # Would need to get from context
        
        # TODO: Implement re-quantization logic
        # This requires accessing the quantizer instance from context
        logger.warning("Re-quantization not yet implemented in phase framework")
