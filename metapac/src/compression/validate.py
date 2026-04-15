"""
Validation utilities for compressed models.

Provides functions to evaluate compressed models on datasets
and compare with baseline models.
"""
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Callable, Tuple, Union, TYPE_CHECKING

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from metapac.src.utils.dataset_repository import load_managed_dataset, resolve_dataset_reference

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:  # pragma: no cover - fallback for runtime when typing symbols unavailable
    PreTrainedTokenizerBase = Any  # type: ignore


def _tensor_elems_bytes(tensors: Iterable[torch.Tensor]) -> int:
    total = 0
    for t in tensors:
        try:
            total += t.numel() * t.element_size()
        except Exception:
            # Fallback in rare cases
            total += int(t.nelement()) * int(getattr(t, 'element_size', lambda: 0)())
    return total


def estimate_model_memory_bytes(model: nn.Module) -> int:
    """Estimate in-memory size of parameters and buffers in bytes.

    This ignores optimizer states and activations; it focuses on the static model
    footprint when loaded.
    """
    params_bytes = _tensor_elems_bytes(model.parameters())
    buffers_bytes = _tensor_elems_bytes(model.buffers())
    return params_bytes + buffers_bytes


def sum_weight_files_bytes(model_dir: Path) -> int:
    """Sum sizes (bytes) of common weight files under a model directory.

    Includes: *.safetensors, *.bin, *.pt, *.pth, and files named like pytorch_model*.bin.
    """
    if not model_dir.exists():
        return 0
    total = 0
    patterns = [
        '*.safetensors',
        '*.bin',
        '*.pt',
        '*.pth',
        'pytorch_model*.bin',
    ]
    for pat in patterns:
        for p in model_dir.rglob(pat):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def _resolve_device(requested: Optional[str]) -> Tuple[str, torch.device]:
    runtime = requested or "auto"
    runtime = runtime.lower()
    if runtime == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.device("mps")
        return "cpu", torch.device("cpu")

    try:
        device = torch.device(runtime)
    except Exception as exc:
        raise ValueError(f"Invalid runtime/device specification: {runtime}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS runtime requested but torch.backends.mps.is_available() is False")
    return str(device), device


def _should_fallback_to_cpu(err: RuntimeError, device_str: str) -> bool:
    if not device_str.startswith("cuda"):
        return False
    msg = str(err).lower()
    fallback_triggers = [
        "no kernel image",
        "cuda error",
        "device-side assert",
        "cublas",
        "cudnn",
    ]
    return any(token in msg for token in fallback_triggers)


def _sync_device(device_str: str) -> None:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_str == "mps" and getattr(torch.backends, "mps", None):
        try:
            torch.mps.synchronize()  # type: ignore[attr-defined]
        except AttributeError:
            pass


def _pre_dequantize_modules(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        fn = getattr(module, "dequantize_weight", None)
        if fn is None:
            continue
        try:
            if getattr(module, "_dequantized_weight", None) is None:
                with torch.no_grad():
                    weight = fn()
                    if getattr(module, "eager_dequant", False):
                        module._dequantized_weight = weight
            count += 1
        except Exception as exc:
            print(f"[validate] pre-dequantize failed for {type(module)}: {exc}", flush=True)
    return count


def _detect_dataset_fields(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    text_field = None
    for candidate in [
        "sentence",
        "text",
        "sentence1",
        "sentence2",
        "article",
        "review",
        "query",
        "premise",
        "hypothesis",
    ]:
        if candidate in sample:
            text_field = candidate
            break
    if text_field is None:
        for key, value in sample.items():
            if isinstance(value, str):
                text_field = key
                break

    label_field = None
    for candidate in ["label", "labels", "target"]:
        if candidate in sample:
            label_field = candidate
            break

    return text_field, label_field


def _tokenize_dataset(
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        dataset_config: str,
        split: str,
        *,
        dataset_source: Optional[Dict[str, Any]],
        dataset_processing: Optional[Dict[str, Any]],
        max_length: int,
        num_proc: Optional[int],
        max_samples: Optional[int]
) -> Tuple[Any, Dict[str, Any]]:
    resolved_name, resolved_config = resolve_dataset_reference(dataset_name, dataset_config)
    print(f"[validate] loading dataset {resolved_name}/{resolved_config} split={split}", flush=True)
    load_start = time.time()
    managed = load_managed_dataset(
        resolved_name,
        resolved_config,
        source_cfg=dataset_source,
        processing_cfg=dataset_processing,
    )
    if split not in managed:
        raise KeyError(f"Requested split '{split}' not found in managed dataset")
    raw_ds = managed[split]
    if max_samples is not None:
        max_samples = min(max_samples, len(raw_ds))
        raw_ds = raw_ds.select(range(max_samples))
    print(f"[validate] dataset loaded in {time.time() - load_start:.1f}s; examples={len(raw_ds)}", flush=True)

    sample = raw_ds[0] if len(raw_ds) > 0 else {}
    text_field, label_field = _detect_dataset_fields(sample)
    if text_field is None:
        raise RuntimeError(
            f"Could not detect a text field in dataset {dataset_name}/{dataset_config} split={split}"
        )

    if label_field is None:
        raw_ds = raw_ds.map(lambda ex: {"label": 0})
        label_field = "label"

    def _tokenize_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        texts = examples[text_field]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

    print(f"[validate] tokenizing with num_proc={num_proc}", flush=True)
    tok_start = time.time()
    if num_proc is None:
        tokenized = raw_ds.map(_tokenize_batch, batched=True)
    else:
        tokenized = raw_ds.map(_tokenize_batch, batched=True, num_proc=num_proc)
    print(f"[validate] tokenization finished in {time.time() - tok_start:.1f}s", flush=True)

    if label_field not in tokenized.column_names:
        tokenized = tokenized.add_column("label", [0] * len(tokenized))

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    metadata = {
        "num_examples": len(tokenized),
        "text_field": text_field,
        "label_field": label_field,
    }
    return tokenized, metadata


def _build_dataloader(dataset: Any, batch_size: int, loader_kwargs: Optional[Dict[str, Any]] = None) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if loader_kwargs:
        kwargs.update(loader_kwargs)
    kwargs["batch_size"] = batch_size
    if kwargs.get("num_workers", 0) == 0:
        kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=False)
        else:
            moved[key] = value
    return moved


def _extract_labels(batch: Dict[str, Any]) -> torch.Tensor:
    if "label" in batch:
        return batch["label"]
    if "labels" in batch:
        return batch["labels"]
    raise KeyError("Batch does not contain 'label' or 'labels'")


def _should_pre_dequantize(device_str: str, override: Optional[bool]) -> bool:
    if override is not None:
        return override
    print("Pre-dequantization is ENABLED — this may take a while…", flush=True)
    return device_str.startswith("cuda")


def _evaluate_model(
        model: nn.Module,
        make_loader: Callable[[], DataLoader],
        *,
        device_str: str,
        warmup_steps: int,
        disable_progress: bool,
        pre_dequantize: bool,
        allow_fallback: bool
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        return _evaluate_model_once(
            model,
            make_loader,
            device_str=device_str,
            warmup_steps=warmup_steps,
            disable_progress=disable_progress,
            pre_dequantize=pre_dequantize,
        )
    except RuntimeError as err:
        if not allow_fallback or not _should_fallback_to_cpu(err, device_str):
            raise
        print(
            f"[validate] Runtime error on {device_str}; retrying on CPU. Error: {err}",
            flush=True,
        )
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _evaluate_model_once(
            model,
            make_loader,
            device_str="cpu",
            warmup_steps=warmup_steps,
            disable_progress=disable_progress,
            pre_dequantize=False,
        )


def _evaluate_model_once(
        model: nn.Module,
        make_loader: Callable[[], DataLoader],
        *,
        device_str: str,
        warmup_steps: int,
        disable_progress: bool,
        pre_dequantize: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    torch_device = torch.device(device_str)
    model.to(torch_device)
    was_training = model.training
    model.eval()

    pre_count = _pre_dequantize_modules(model) if pre_dequantize else 0

    warmup_batches = 0
    if warmup_steps > 0:
        warmup_loader = make_loader()
        with torch.no_grad():
            warmup_iter = iter(warmup_loader)
            for _ in range(warmup_steps):
                try:
                    batch = next(warmup_iter)
                except StopIteration:
                    break
                moved = _move_batch_to_device(batch, torch_device)
                inputs = {
                    key: value
                    for key, value in moved.items()
                    if key not in {"label", "labels"}
                }
                _ = model(**inputs)
                warmup_batches += 1
                _sync_device(device_str)
        del warmup_loader

    loader = make_loader()
    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None

    iterator = loader
    if not disable_progress:
        iterator = tqdm(loader, desc=f"Evaluating [{device_str}]", total=total_batches)

    correct = 0
    total = 0
    batches = 0
    eval_start = time.time()

    with torch.no_grad():
        for batch in iterator:
            moved = _move_batch_to_device(batch, torch_device)
            labels = _extract_labels(moved)
            inputs = {
                key: value
                for key, value in moved.items()
                if key not in {"label", "labels"}
            }
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            batches += 1
            _sync_device(device_str)

    elapsed = time.time() - eval_start
    accuracy = correct / total if total else 0.0
    examples_per_sec = (total / elapsed) if elapsed > 0 else None

    if was_training:
        model.train()

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "batches": batches,
        "elapsed_sec": elapsed,
        "examples_per_sec": examples_per_sec,
    }
    details = {
        "used_device": device_str,
        "pre_dequantized_layers": pre_count,
        "warmup_batches": warmup_batches,
    }
    return metrics, details


def _has_quantized_modules(model: nn.Module) -> bool:
    for m in model.modules():
        # Adjust this if your quantized class name differs.
        if m.__class__.__name__.startswith("CUDAQuantized"):
            return True
    return False


def validate_model(
        subject_model: nn.Module,
        subject_tokenizer: Optional[PreTrainedTokenizerBase],
        *,
        dataset: str = "glue",
        dataset_config: str = "sst2",
        split: str = "validation",
        batch_size: int = 32,
        max_length: int = 128,
        runtime: str = "auto",
        device: Optional[str] = None,
        warmup_steps: int = 2,
        disable_progress: bool = False,
        pre_dequantize: Optional[bool] = None,
        num_proc: Optional[int] = None,
        max_samples: Optional[int] = None,
        dataset_source: Optional[Dict[str, Any]] = None,
        dataset_processing: Optional[Dict[str, Any]] = None,
        subject_name: str = "subject",
        subject_path: Optional[Union[str, Path]] = None,
        subject_dataset: Optional[Any] = None,
        baseline_model: Optional[nn.Module] = None,
        baseline_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        baseline_name: str = "baseline",
        baseline_path: Optional[Union[str, Path]] = None,
        baseline_dataset: Optional[Any] = None,
        allow_runtime_fallback: bool = True,
        loader_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a model (and optional baseline) on a HuggingFace dataset.

    Args:
        subject_model: Model to evaluate.
        subject_tokenizer: Tokenizer for the subject model. Required unless ``subject_dataset`` provided.
        dataset: Dataset name for ``datasets.load_dataset``.
        dataset_config: Dataset configuration.
        split: Dataset split to use.
        batch_size: Batch size for evaluation.
        max_length: Maximum sequence length during tokenization.
        runtime: Runtime preference (``"auto"``/``"cuda"``/``"cpu"``/``"mps"``).
        device: Legacy alias for ``runtime``; if provided it overrides ``runtime``.
        warmup_steps: Number of warmup batches before measurement.
        disable_progress: Disable tqdm progress bars.
        pre_dequantize: Override eager pre-dequantisation flag. ``None`` keeps heuristic.
        num_proc: Number of tokenizer workers. Defaults to 1 on Windows, ``datasets`` default elsewhere.
        max_samples: Optional cap on number of examples.
        subject_name: Label for the subject entry in the output dictionary.
        subject_path: Optional filesystem path to compute on-disk sizes.
        subject_dataset: Pre-tokenized dataset to reuse instead of loading via HuggingFace.
        baseline_model: Optional baseline model for comparison.
        baseline_tokenizer: Tokenizer for the baseline; defaults to ``subject_tokenizer`` when omitted.
        baseline_name: Label for the baseline entry in the output dictionary.
        baseline_path: Optional baseline path for size computation.
        baseline_dataset: Pre-tokenized dataset for the baseline model.
        allow_runtime_fallback: Whether to retry on CPU after CUDA runtime failures.
        loader_kwargs: Extra ``DataLoader`` keyword arguments.

    Returns:
        Dictionary with ``subject`` (and optional ``baseline``) metrics plus comparison details.
    """

    effective_runtime = device or runtime or "auto"
    requested_device_str, requested_device = _resolve_device(effective_runtime)

    loader_kwargs = loader_kwargs or {}
    auto_num_proc = None  # Disable multiprocessing by default (avoid Windows issues)
    num_proc = num_proc if num_proc is not None else auto_num_proc

    dataset_info: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "batch_size": batch_size,
        "max_length": max_length,
    }

    if subject_dataset is not None:
        tokenized_subject = subject_dataset
        dataset_info["num_examples"] = len(subject_dataset)
    else:
        if subject_tokenizer is None:
            raise ValueError("subject_tokenizer is required when subject_dataset is not provided")
        tokenized_subject, meta = _tokenize_dataset(
            subject_tokenizer,
            dataset,
            dataset_config,
            split,
            dataset_source=dataset_source,
            dataset_processing=dataset_processing,
            max_length=max_length,
            num_proc=num_proc,
            max_samples=max_samples,
        )
        dataset_info.update(meta)

    subject_loader_factory = lambda: _build_dataloader(tokenized_subject, batch_size, loader_kwargs)
    is_quantized_runtime = _has_quantized_modules(subject_model)
    if is_quantized_runtime:
        subject_pre_dequant = False
        print("[validate] Detected quantized runtime; skipping pre-dequant.", flush=True)
    else:
        subject_pre_dequant = _should_pre_dequantize(requested_device_str, pre_dequantize)
    subject_metrics, subject_details = _evaluate_model(
        subject_model,
        subject_loader_factory,
        device_str=requested_device_str,
        warmup_steps=warmup_steps,
        disable_progress=disable_progress,
        pre_dequantize=subject_pre_dequant,
        allow_fallback=allow_runtime_fallback,
    )

    subject_sizes = {
        "param_buffer_bytes": estimate_model_memory_bytes(subject_model),
    }
    if subject_path:
        subject_path_obj = Path(subject_path)
        if subject_path_obj.exists():
            subject_sizes["file_bytes"] = sum_weight_files_bytes(subject_path_obj)

    subject_result = {
        "name": subject_name,
        **subject_metrics,
        "device": subject_details["used_device"],
        "pre_dequantized_layers": subject_details["pre_dequantized_layers"],
        "warmup_batches": subject_details["warmup_batches"],
        "sizes": subject_sizes,
    }
    if subject_path:
        subject_result["path"] = str(subject_path)

    print(
        f"[validate] {subject_name} accuracy: {subject_metrics['accuracy']:.4f} "
        f"({subject_metrics['accuracy'] * 100:.2f}%) on {subject_metrics['total']} examples",
        flush=True,
    )

    results: Dict[str, Any] = {
        "subject": subject_result,
        "dataset": dataset_info,
        "runtime": {
            "requested": effective_runtime,
            "subject_used": subject_details["used_device"],
        },
    }

    if baseline_model is not None:
        baseline_tokenizer = baseline_tokenizer or subject_tokenizer
        if baseline_dataset is not None:
            tokenized_baseline = baseline_dataset
        elif baseline_tokenizer is subject_tokenizer and subject_dataset is None:
            tokenized_baseline = tokenized_subject
        else:
            if baseline_tokenizer is None:
                raise ValueError("baseline_tokenizer is required when baseline_dataset is not provided")
            tokenized_baseline, _ = _tokenize_dataset(
                baseline_tokenizer,
                dataset,
                dataset_config,
                split,
                max_length=max_length,
                num_proc=num_proc,
                max_samples=max_samples,
            )

        baseline_loader_factory = lambda: _build_dataloader(tokenized_baseline, batch_size, loader_kwargs)
        baseline_metrics, baseline_details = _evaluate_model(
            baseline_model,
            baseline_loader_factory,
            device_str=requested_device_str,
            warmup_steps=warmup_steps,
            disable_progress=disable_progress,
            pre_dequantize=_should_pre_dequantize(requested_device_str, pre_dequantize),
            allow_fallback=allow_runtime_fallback,
        )

        baseline_sizes = {
            "param_buffer_bytes": estimate_model_memory_bytes(baseline_model),
        }
        if baseline_path:
            baseline_path_obj = Path(baseline_path)
            if baseline_path_obj.exists():
                baseline_sizes["file_bytes"] = sum_weight_files_bytes(baseline_path_obj)

        baseline_result = {
            "name": baseline_name,
            **baseline_metrics,
            "device": baseline_details["used_device"],
            "pre_dequantized_layers": baseline_details["pre_dequantized_layers"],
            "warmup_batches": baseline_details["warmup_batches"],
            "sizes": baseline_sizes,
        }
        if baseline_path:
            baseline_result["path"] = str(baseline_path)

        print(
            f"[validate] {baseline_name} accuracy: {baseline_metrics['accuracy']:.4f} "
            f"({baseline_metrics['accuracy'] * 100:.2f}%) on {baseline_metrics['total']} examples",
            flush=True,
        )

        accuracy_drop = baseline_metrics["accuracy"] - subject_metrics["accuracy"]
        baseline_accuracy = baseline_metrics["accuracy"]
        accuracy_drop_pct = (
            (accuracy_drop / baseline_accuracy * 100)
            if baseline_accuracy > 0
            else 0
        )

        size_comparison: Dict[str, Any] = {}
        if "file_bytes" in subject_sizes and "file_bytes" in baseline_sizes:
            file_delta = baseline_sizes["file_bytes"] - subject_sizes["file_bytes"]
            size_comparison["file_bytes_reduction"] = file_delta
            size_comparison["file_bytes_reduction_pct"] = (
                (file_delta / baseline_sizes["file_bytes"] * 100)
                if baseline_sizes["file_bytes"]
                else 0
            )
        if (
                "param_buffer_bytes" in subject_sizes
                and "param_buffer_bytes" in baseline_sizes
        ):
            mem_delta = baseline_sizes["param_buffer_bytes"] - subject_sizes["param_buffer_bytes"]
            size_comparison["param_buffer_bytes_reduction"] = mem_delta
            size_comparison["param_buffer_bytes_reduction_pct"] = (
                (mem_delta / baseline_sizes["param_buffer_bytes"] * 100)
                if baseline_sizes["param_buffer_bytes"]
                else 0
            )

        results.update(
            {
                "baseline": baseline_result,
                "comparison": {
                    "accuracy_drop": accuracy_drop,
                    "accuracy_drop_pct": accuracy_drop_pct,
                    **size_comparison,
                },
            }
        )
        results["runtime"]["baseline_used"] = baseline_details["used_device"]

        print(
            f"[validate] accuracy delta (baseline - {subject_name}): {accuracy_drop:.4f} "
            f"({accuracy_drop_pct:.2f}%)",
            flush=True,
        )

    return results


# Backward compatibility helpers -------------------------------------------------

def evaluate_on_dataset(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward compatible wrapper for legacy callers.

    Deprecated: use :func:`validate_model` directly.
    """

    result = validate_model(*args, **kwargs)
    return result["subject"]


def validate_compressed_model(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Backward compatible wrapper kept for legacy imports.

    Delegates to :func:`validate_model` with newer semantics.
    """

    raise RuntimeError(
        "validate_compressed_model has been replaced by validate_model. "
        "Update callers to use validate_model(subject_model=..., baseline_model=..., ...)."
    )
