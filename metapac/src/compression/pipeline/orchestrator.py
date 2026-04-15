"""Compression pipeline orchestrator.

Coordinates the execution of compression phases in sequence:
1. Preparation: Feature extraction, importance scoring, zoning
2. Pruning: Remove unimportant structures
3. Quantization: Reduce bit-width based on importance
4. Fine-Tuning: Recover accuracy after compression
5. Export: Save compressed model and metadata
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import torch.nn as nn

from .phase_base import PhaseContext, CompressionPhase
from .config_manager import load_strategy_defaults, merge_with_defaults
from ..phases.preparation import load_meta_predictor_checkpoint, extract_parameter_features, compute_importance_scores, rank_and_partition_parameters
from ..phases.pruning_phase import PruningPhase
from ..phases.quantization_phase import QuantizationPhase
from ..phases.fine_tuning import FineTuningPhase
from ..phases.export import ExportPhase
from ..utils.model_loading import load_target_model, snapshot_state_dict_cpu, state_dict_change_stats
from ..utils.checkpoint import select_checkpoint
from ...utils.logging_utils import get_logger, log_phase_header, log_section, log_metric

logger = get_logger(__name__)


class CompressionPipeline:
    """Main compression pipeline orchestrator.
    
    Coordinates phases:
    - Preparation (feature extraction, scoring, zoning)
    - Compression (pruning, quantization)
    - Recovery (fine-tuning)
    - Export (model saving)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration.
        
        Args:
            config: Full compression configuration
        """
        self.config = self._prepare_config(config)
        self.compression_cfg = self.config.get('compression', {})
        
        # Initialize phases
        self.phases = self._initialize_phases()
    
    def _prepare_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults."""
        defaults = load_strategy_defaults()
        default_compression = defaults.get("compression", {})
        
        # Merge compression config
        compression_cfg = merge_with_defaults(
            user_config.get("compression", {}), 
            default_compression
        )
        
        # Apply checkpoint selector
        target_model = compression_cfg.get("target_model")
        checkpoint_selector_cfg = compression_cfg.get("checkpoint_selector", {}) or {}
        
        selected_checkpoint = select_checkpoint(
            model_ref=target_model,
            mode=checkpoint_selector_cfg.get("mode"),
            exact_step=checkpoint_selector_cfg.get("exact_step"),
        )
        
        if selected_checkpoint != target_model:
            logger.info(
                f"Checkpoint selector resolved: {target_model} -> {selected_checkpoint}"
            )
            compression_cfg["target_model"] = selected_checkpoint
            
            # Align baseline model
            if compression_cfg.get("baseline_model_config") is None:
                compression_cfg["baseline_model_config"] = selected_checkpoint
        
        user_config['compression'] = compression_cfg
        return user_config
    
    def _initialize_phases(self) -> list[CompressionPhase]:
        """Initialize compression phases."""
        phases = []
        
        # Pruning phase
        pruning_cfg = self.compression_cfg.get('pruning', {})
        if pruning_cfg.get('enabled', False):
            phases.append(PruningPhase(pruning_cfg))
        
        # Quantization phase
        quantization_cfg = self.compression_cfg.get('quantization', {})
        if quantization_cfg.get('enabled', True):
            phases.append(QuantizationPhase(quantization_cfg))
        
        # Fine-tuning phase
        fine_tuning_cfg = self.compression_cfg.get('fine_tuning', {})
        if fine_tuning_cfg.get('enabled', False):
            phases.append(FineTuningPhase(fine_tuning_cfg))
        
        # Export phase (always enabled)
        export_cfg = quantization_cfg  # Export config is in quantization section
        phases.append(ExportPhase(export_cfg))
        
        return phases
    
    def run(self) -> int:
        """Run the complete compression pipeline.
        
        Returns:
            0 on success, 1 on failure
        """
        try:
            # Phase 1: Preparation (feature extraction, scoring, zoning)
            context = self._run_preparation()
            
            # Snapshot baseline for change detection
            state_before = snapshot_state_dict_cpu(context.model)
            
            # Phase 2-5: Compression, fine-tuning, export
            for phase in self.phases:
                context = phase.run(context)
            
            # Validate weight changes
            if not self._validate_weight_changes(context.model, state_before):
                return 1
            
            logger.info("✓ Compression pipeline completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_preparation(self) -> PhaseContext:
        """Run preparation phase (feature extraction, scoring, zoning)."""
        log_phase_header(logger, "1: Load Meta-Predictor & Compute Importance")
        
        target_model = self.compression_cfg.get('target_model')
        meta_checkpoint = self.compression_cfg.get('meta_checkpoint')
        zones_config = self.compression_cfg.get('zones', {})
        zone_assignment = self.compression_cfg.get('zone_assignment', {})
        output_dir = self.config.get('output_dir', 'output/compression')
        
        # Load meta-predictor
        model, imputer, scaler, feature_names, target_name, task_type = \
            load_meta_predictor_checkpoint(meta_checkpoint)
        
        logger.info(f"Meta-predictor loaded: task={task_type}, features={len(feature_names)}")
        
        # Wrap model with preprocessing if needed
        if imputer is not None and scaler is not None:
            from ..phases.preparation import create_preprocessed_pipeline
            pipeline = create_preprocessed_pipeline(model, imputer, scaler)
        else:
            pipeline = model
        
        # Extract features
        features_df = extract_parameter_features(target_model, feature_names)
        
        # Compute importance scores
        importance_df = compute_importance_scores(
            pipeline,
            features_df,
            feature_names
        )
        
        # Rank and partition into zones
        log_phase_header(logger, "2: Rank & Partition into Zones")
        ranked_df = rank_and_partition_parameters(
            importance_df,
            zones_config=zones_config,
            zone_assignment_cfg=zone_assignment
        )
        
        # Load target model
        log_phase_header(logger, "3: Apply Zone-Specific Compression")
        target_model_instance = load_target_model(target_model)
        
        # Log baseline statistics
        log_section(logger, "Baseline Model")
        baseline_size_mb = sum(p.numel() * p.element_size() for p in target_model_instance.parameters()) / (1024 * 1024)
        log_metric(logger, "Size", f"{baseline_size_mb:.2f}", "MB")
        log_metric(logger, "Parameters", sum(p.numel() for p in target_model_instance.parameters()))
        
        # Build compression plan
        plan = ranked_df.set_index('parameter_name')['action'].to_dict()
        importance_rankings = ranked_df.set_index('parameter_name')['importance_score'].to_dict()
        target_bits_map = ranked_df.set_index('parameter_name')['target_bits'].to_dict()
        
        # Normalize importance scores for quantize zone
        quantize_params = ranked_df[ranked_df['action'] == 'quantize']
        if len(quantize_params) > 0:
            min_score = quantize_params['importance_score'].min()
            max_score = quantize_params['importance_score'].max()
            if max_score > min_score:
                for param_name in quantize_params['parameter_name']:
                    score = importance_rankings[param_name]
                    importance_rankings[param_name] = (score - min_score) / (max_score - min_score)
            else:
                for param_name in quantize_params['parameter_name']:
                    importance_rankings[param_name] = 0.5
        
        # Create context
        context = PhaseContext(
            model=target_model_instance,
            config=self.config,
            output_path=Path(output_dir),
            metadata={
                'preparation': {
                    'features_extracted': len(features_df),
                    'importance_computed': len(importance_df),
                    'parameters_ranked': len(ranked_df)
                }
            },
            ranked_df=ranked_df,
            plan=plan,
            importance_rankings=importance_rankings,
            target_bits_map=target_bits_map
        )
        
        return context
    
    def _validate_weight_changes(self, model: nn.Module, state_before: Dict) -> bool:
        """Validate that compression actually changed model weights."""
        require_change = self.compression_cfg.get('require_weight_change_for_success', False)
        quantization_enabled = self.compression_cfg.get('quantization', {}).get('enabled', True)
        
        if not require_change or not quantization_enabled:
            return True
        
        state_after = snapshot_state_dict_cpu(model)
        stats = state_dict_change_stats(state_before, state_after, atol=0.0)
        
        logger.info(
            f"Weight-change check: changed_tensors={stats['changed_tensors']}/"
            f"{stats['shared_tensors']}, max_abs_diff={stats['max_abs_diff']:.6g}"
        )
        
        if stats["changed_tensors"] == 0:
            logger.error(
                "Compression produced no weight changes (changed_tensors=0). "
                "Failing run to avoid misleading baseline==compressed results. "
                "Set compression.require_weight_change_for_success=false to allow this."
            )
            return False
        
        return True


def run_compression(cfg: Dict[str, Any]) -> int:
    """Main entry point for compression pipeline.
    
    Args:
        cfg: Compression configuration dictionary
        
    Returns:
        0 on success, 1 on failure
    """
    pipeline = CompressionPipeline(cfg)
    return pipeline.run()
