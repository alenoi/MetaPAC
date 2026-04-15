"""
Fine-tuning script for pruned transformer models.

This script fine-tunes a pruned model to recover accuracy after structured pruning.
It supports both DistilBERT and other transformer architectures.

Usage:
    python -m metapac.src.compression.fine_tune --config config.yaml
"""
from __future__ import annotations

# Set offline mode BEFORE any HuggingFace imports
import os

# os.environ['HF_DATASETS_OFFLINE'] = '1'  # Disabled - allow online dataset loading
# os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Disabled - allow online model loading
# Disable PyTorch threading to prevent CPU hangs
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from metapac.src.utils.hf_sources import (
    load_sequence_classification_model_from_source,
    load_tokenizer_from_source,
)
from metapac.src.utils.dataset_repository import load_managed_dataset, resolve_dataset_reference

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
except ImportError:
    print("Warning: transformers and datasets not available. Fine-tuning will be limited.")

logger = logging.getLogger(__name__)


def _resolve_runtime_device(requested: Optional[str]) -> str:
    """Resolve runtime device safely with graceful fallback.

    - "auto" prefers CUDA, then MPS, then CPU.
    - Explicit "cuda" falls back to CPU if CUDA is unavailable.
    - Explicit "mps" falls back to CPU if MPS is unavailable.
    """
    runtime = str(requested or "auto").lower()

    if runtime == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if runtime.startswith("cuda"):
        if torch.cuda.is_available():
            return runtime
        logger.warning("Requested CUDA fine-tuning device, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"

    if runtime.startswith("mps"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return runtime
        logger.warning("Requested MPS fine-tuning device, but MPS is unavailable. Falling back to CPU.")
        return "cpu"

    return runtime


def _resolve_model_reference(model_ref: Optional[str]) -> Optional[str]:
    """Resolve local model references to absolute paths when possible.

    Keeps non-local references (e.g., HF hub IDs) unchanged.
    """
    if not model_ref:
        return model_ref

    candidate = Path(model_ref)
    if candidate.exists():
        return str(candidate.resolve())
    return model_ref


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge Distillation Loss.
    
    Based on: Hinton et al. (2014) "Distilling the Knowledge in a Neural Network"
    
    Combines KL divergence between teacher and student distributions with cross-entropy loss.
    
    Args:
        temperature: Temperature for softmax scaling (higher = softer distribution)
        alpha: Weight between KD loss and CE loss (0.5 = 50-50 split)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        L = alpha * L_KD + (1 - alpha) * L_CE
        
        where L_KD = T^2 * KL(softmax(teacher/T), softmax(student/T))
        
        Args:
            student_logits: Student model outputs (batch, num_classes)
            teacher_logits: Teacher model outputs (batch, num_classes)
            labels: Ground truth labels (batch,)
        
        Returns:
            Combined loss
        """
        # KD loss: match probability distributions
        student_probs = torch.nn.functional.log_softmax(
            student_logits / self.temperature, dim=1
        )
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / self.temperature, dim=1
        )
        kd_loss = self.kd_loss(student_probs, teacher_probs)

        # Scale KD loss by temperature squared (from original paper)
        kd_loss = kd_loss * (self.temperature ** 2)

        # CE loss: match ground truth
        ce_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        combined_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        return combined_loss


class FineTuner:
    """Fine-tuning manager for pruned models.
    
    Supports optional Knowledge Distillation with a teacher model.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: Dict[str, Any],
            teacher_model: Optional[nn.Module] = None
    ):
        """Initialize fine-tuner.
        
        Args:
            model: Pruned model to fine-tune.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Fine-tuning configuration.
            teacher_model: Optional teacher model for knowledge distillation.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.teacher_model = teacher_model

        # Training hyperparameters with explicit type conversion
        self.num_epochs = int(config.get('num_epochs', 5))
        self.learning_rate = float(config.get('learning_rate', 2e-5))
        self.weight_decay = float(config.get('weight_decay', 0.01))
        self.warmup_ratio = float(config.get('warmup_ratio', 0.1))
        self.gradient_clip = float(config.get('gradient_clip', 1.0))

        # Device (robust to unavailable CUDA builds)
        self.device = _resolve_runtime_device(config.get('device', 'auto'))
        self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()  # Teacher is not trained

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Loss function
        # KD config from training config (where it's passed), or defaults
        kd_config = config.get('distillation', {})
        kd_enabled = kd_config.get('enabled', False) and teacher_model is not None

        if kd_enabled:
            temperature = float(kd_config.get('temperature', 4.0))
            alpha = float(kd_config.get('alpha', 0.5))
            self.criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
            self.use_kd = True
            logger.info(f"[FINE-TUNE] Knowledge Distillation enabled (T={temperature}, alpha={alpha})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.use_kd = False
            logger.info(f"[FINE-TUNE] Using standard CrossEntropyLoss (no KD)")

        # Tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        # Early stopping configuration
        early_stop_config = config.get('early_stopping', {})
        self.early_stop_enabled = early_stop_config.get('enabled', False)
        self.early_stop_patience = early_stop_config.get('patience', 3)
        self.early_stop_min_delta = early_stop_config.get('min_delta', 0.001)
        self.early_stop_counter = 0
        
        if self.early_stop_enabled:
            logger.info(f"[FINE-TUNE] Early stopping enabled (patience={self.early_stop_patience}, min_delta={self.early_stop_min_delta})")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Optionally uses Knowledge Distillation if teacher model is available.
        
        Returns:
            Dict with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass - student
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Compute loss
            if self.use_kd and self.teacher_model is not None:
                # Knowledge Distillation: get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs

                # KD loss
                loss = self.criterion(student_logits, teacher_logits, labels)
            else:
                # Standard CE loss
                loss = self.criterion(student_logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(student_logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct / total:.4f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.
        
        Returns:
            Dict with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Use CE loss for evaluation (not KD)
        ce_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Use CE loss for evaluation
                loss = ce_criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def fine_tune(self) -> Dict[str, Any]:
        """Run fine-tuning loop.
        
        Returns:
            Dict with fine-tuning results.
        """
        logger.info("=" * 60)
        logger.info("FINE-TUNING PRUNED MODEL")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Weight decay: {self.weight_decay}")
        logger.info(f"Device: {self.device}")

        history = []

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            history.append(metrics)

            # Log
            logger.info(
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )

            # Save best model and check early stopping
            if val_metrics['val_accuracy'] > self.best_val_accuracy + self.early_stop_min_delta:
                self.best_val_accuracy = val_metrics['val_accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.early_stop_counter = 0  # Reset counter
                logger.info(f"✓ New best model! Val Acc: {self.best_val_accuracy:.4f}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_enabled:
                    logger.info(f"No improvement for {self.early_stop_counter} epoch(s)")
                    
                    # Check early stopping
                    if self.early_stop_counter >= self.early_stop_patience:
                        logger.info(f"Early stopping triggered! No improvement for {self.early_stop_patience} epochs.")
                        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
                        break  # Exit training loop

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val acc: {self.best_val_accuracy:.4f}")

        logger.info("=" * 60)
        logger.info("FINE-TUNING COMPLETE")
        logger.info("=" * 60)

        return {
            'history': history,
            'best_val_accuracy': self.best_val_accuracy,
            'final_val_accuracy': history[-1]['val_accuracy'] if history else 0.0
        }


def load_pruned_model(checkpoint_path: str, base_model: str = None) -> nn.Module:
    """Load pruned model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        base_model: Base model architecture to use (for state dict loading).
    
    Returns:
        Loaded model.
    """
    checkpoint_path = _resolve_model_reference(checkpoint_path)
    base_model = _resolve_model_reference(base_model)
    logger.info(f"Loading pruned model from: {checkpoint_path}")

    # Try to load as transformers model
    try:
        model = load_sequence_classification_model_from_source(checkpoint_path, {"mode": "local"})
        logger.info("Loaded model using transformers AutoModel")
    except Exception as e:
        # Fallback: load state dict directly
        logger.warning(f"Could not load with AutoModel: {e}")
        logger.info("Attempting to load state dict...")

        checkpoint_dir = Path(checkpoint_path)
        state_candidates = [
            checkpoint_dir / "model_state.pt",
            checkpoint_dir / "pytorch_model.bin",
            checkpoint_dir / "model.safetensors",
            checkpoint_dir / "model.pt",
        ]
        state_path = next((p for p in state_candidates if p.exists()), None)
        compression_summary_path = checkpoint_dir / "compression_summary.json"

        if state_path is None:
            raise FileNotFoundError(
                f"Model state not found in checkpoint dir: {checkpoint_dir}. "
                f"Checked: {[str(p.name) for p in state_candidates]}"
            )

        # Try to get base model from compression summary
        if base_model is None and compression_summary_path.exists():
            import json
            with open(compression_summary_path) as f:
                summary = json.load(f)
                base_model = summary.get('target_model')
                base_model = _resolve_model_reference(base_model)
                logger.info(f"Found base model in compression summary: {base_model}")

        if base_model is None:
            raise ValueError(
                "Cannot load state dict without knowing base model architecture. "
                "Please provide base_model parameter or ensure compression_summary.json exists."
            )

        # Load base model architecture
        logger.info(f"Loading base model architecture from: {base_model}")
        model = load_sequence_classification_model_from_source(base_model)

        # Load compressed state dict
        logger.info(f"Loading state dict from: {state_path}")
        if state_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(str(state_path), device='cpu')
        else:
            state_dict = torch.load(state_path, map_location='cpu')

        # Load state dict with strict=False to handle pruned parameters
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading state dict: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys[:5]}...")

        logger.info("Successfully loaded pruned model from state dict")

    return model


def prepare_data_loaders(
        tokenizer,
        config: Dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders.
    
    Args:
        tokenizer: Tokenizer for encoding text.
        config: Data configuration.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset_name = config.get('dataset', 'glue')
    dataset_config = config.get('dataset_config', 'sst2')
    dataset_source = config.get('source')
    max_length = int(config.get('max_length', 512))
    batch_size = int(config.get('batch_size', 32))

    resolved_name, resolved_config = resolve_dataset_reference(dataset_name, dataset_config)
    logger.info(f"Loading dataset: {resolved_name}/{resolved_config}")
    try:
        dataset = load_managed_dataset(
            resolved_name,
            resolved_config,
            source_cfg=dataset_source,
            processing_cfg=config,
        )
    except Exception as e:
        logger.error(f"Failed to load managed dataset: {e}")
        raise

    # Determine text fields (dataset-specific)
    train_columns = list(dataset['train'].column_names)

    if 'sentence' in train_columns:
        text_fields = ['sentence']
    elif 'text' in train_columns:
        text_fields = ['text']
    elif 'sentence1' in train_columns and 'sentence2' in train_columns:
        text_fields = ['sentence1', 'sentence2']
    elif 'premise' in train_columns and 'hypothesis' in train_columns:
        text_fields = ['premise', 'hypothesis']
    else:
        # Fallback: first non-label textual column
        candidate = [c for c in train_columns if c != 'label']
        if not candidate:
            raise KeyError("No text column found in dataset")
        text_fields = [candidate[0]]

    logger.info(f"Using text fields for tokenization: {text_fields}")

    # Tokenize with dynamic padding (more memory efficient)
    def tokenize_function(examples):
        if len(text_fields) == 1:
            return tokenizer(
                examples[text_fields[0]],
                padding=False,  # Dynamic padding in DataLoader
                truncation=True,
                max_length=max_length
            )
        return tokenizer(
            examples[text_fields[0]],
            examples[text_fields[1]],
            padding=False,
            truncation=True,
            max_length=max_length
        )

    # Get columns to remove (everything except label)
    columns_to_remove = [col for col in dataset['train'].column_names if col != 'label']

    logger.info("Tokenizing dataset (this may take a few minutes)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # Process in smaller chunks
        remove_columns=columns_to_remove,
        desc="Tokenizing"
    )

    logger.info("Renaming columns and setting format...")
    # Rename label column
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
    tokenized_dataset.set_format('torch')

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if 'validation' not in tokenized_dataset:
        raise ValueError(
            "Managed fine-tuning dataset does not provide a validation split. "
            "Configure split_strategy/ratios so the dataset repository materializes one."
        )

    # Optional sample caps for faster/controlled fine-tuning
    train_max_samples = config.get('train_max_samples')
    if train_max_samples is not None:
        train_max_samples = int(train_max_samples)
        if train_max_samples > 0:
            n_train = min(train_max_samples, len(tokenized_dataset['train']))
            tokenized_dataset['train'] = tokenized_dataset['train'].select(range(n_train))
            logger.info(f"Applied train_max_samples cap: {n_train}")

    validation_max_samples = config.get('validation_max_samples')
    if validation_max_samples is not None:
        validation_max_samples = int(validation_max_samples)
        if validation_max_samples > 0:
            n_val = min(validation_max_samples, len(tokenized_dataset['validation']))
            tokenized_dataset['validation'] = tokenized_dataset['validation'].select(range(n_val))
            logger.info(f"Applied validation_max_samples cap: {n_val}")

    # Create data loaders with dynamic padding
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        tokenized_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        tokenized_dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    logger.info(f"Train samples: {len(tokenized_dataset['train'])}")
    logger.info(f"Val samples: {len(tokenized_dataset['validation'])}")

    return train_loader, val_loader


def run_fine_tuning(config: Dict[str, Any]) -> int:
    """Run fine-tuning pipeline programmatically.
    
    Supports optional Knowledge Distillation with teacher model.
    
    Args:
        config: Fine-tuning configuration dictionary with keys:
            - model_checkpoint: Path to pruned model
            - output_dir: Output directory for fine-tuned model
            - data: Data configuration (dataset, batch_size, etc.)
            - training: Training configuration (epochs, lr, etc.)
            - distillation: Optional KD config with teacher_model, temperature, alpha
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Extract configs
        model_checkpoint = config['model_checkpoint']
        output_dir = Path(config.get('output_dir', 'fine_tuned_model'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine base model for tokenizer
        base_model_for_tokenizer = model_checkpoint
        compression_summary_path = Path(model_checkpoint) / "compression_summary.json"
        if compression_summary_path.exists():
            import json
            with open(compression_summary_path) as f:
                summary = json.load(f)
                base_model_for_tokenizer = summary.get('target_model', model_checkpoint)
                logger.info(f"Using base model for tokenizer: {base_model_for_tokenizer}")

        base_model_for_tokenizer = _resolve_model_reference(base_model_for_tokenizer)
        model_source = config.get('model_source')

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = load_pruned_model(model_checkpoint, base_model=base_model_for_tokenizer)
        tokenizer = load_tokenizer_from_source(base_model_for_tokenizer, model_source)

        # Load teacher model for Knowledge Distillation (if enabled)
        teacher_model = None
        kd_config = config.get('distillation', {})
        logger.info(
            f"[FINE-TUNE] Distillation config: enabled={kd_config.get('enabled', False)}, teacher_path={kd_config.get('teacher_model', 'None')}")
        if kd_config.get('enabled', False):
            teacher_model_path = kd_config.get('teacher_model', None)
            if teacher_model_path:
                try:
                    teacher_model_path = _resolve_model_reference(teacher_model_path)
                    logger.info(f"Loading teacher model for Knowledge Distillation from: {teacher_model_path}")
                    teacher_source = kd_config.get('teacher_source')
                    teacher_model = load_sequence_classification_model_from_source(
                        str(teacher_model_path),
                        teacher_source,
                        device_map='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    logger.info(f"Teacher model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load teacher model: {e}")
                    teacher_model = None

        # Prepare data
        logger.info("Preparing data loaders...")
        train_loader, val_loader = prepare_data_loaders(
            tokenizer,
            config.get('data', {})
        )

        # Merge configs for FineTuner (training + distillation)
        ft_all_config = config.get('training', {}).copy()
        ft_all_config['distillation'] = config.get('distillation', {})

        # Create fine-tuner with teacher model for KD
        fine_tuner = FineTuner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=ft_all_config,  # Now includes distillation config
            teacher_model=teacher_model  # Optional KD
        )

        # Fine-tune
        results = fine_tuner.fine_tune()

        # Save fine-tuned model
        logger.info(f"Saving fine-tuned model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save results
        import json
        results_path = output_dir / "fine_tune_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to: {results_path}")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Final validation accuracy: {results['final_val_accuracy']:.4f}")
        logger.info(f"{'=' * 60}")

        return 0

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main fine-tuning entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune pruned transformer model")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to fine-tuning configuration YAML'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to pruned model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for fine-tuned model (overrides config)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.model:
        config['model_checkpoint'] = args.model
    if args.output:
        config['output_dir'] = args.output

    # Run fine-tuning
    return run_fine_tuning(config)


if __name__ == '__main__':
    sys.exit(main())
