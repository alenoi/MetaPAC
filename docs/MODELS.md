# Model Support

MetaPAC 0.2.0 uses a shared sequence-classification handling layer for model loading, baseline fine-tuning, and compression preparation.

## Shared Handling

The active pipeline no longer depends on per-target wrapper scripts for the public workflow.

Instead, it uses shared components for:

- model source resolution through `metapac.src.utils.hf_sources`
- tokenizer loading and baseline training through `metapac.src.model_handlers`
- Hugging Face model construction through `AutoConfig` and `AutoModelForSequenceClassification`
- architecture-specific hook and module naming through `metapac.src.model_profiles`

This shared path applies across all supported model families in the current repository scope.

## Supported Model Families

The active model profiles recognize these families:

- DistilBERT
- BERT
- RoBERTa
- GPT-2, including DistilGPT2-style checkpoints
- Qwen2, Qwen2.5, and Qwen3 sequence-classification checkpoints

The handler registry also includes an auto fallback for Hugging Face sequence-classification models that do not need extra family-specific overrides.

## Curated Example Models

The retained public configs currently provide ready-to-run examples for:

- `distilbert-base-uncased` on SST-2
- `distilgpt2` on IMDB
- `Qwen/Qwen3-0.6B` on WOS

Those are the model references currently documented and kept as the main example entrypoints.

## Model Source Workflows

Model loading is resolved through `metapac.src.utils.hf_sources.resolve_model_source`.

Common patterns:

- standard Hugging Face reference loading from `pretrained_name`
- cache-only execution with `local_files_only: true`
- forced local loading with `mode: local` and an explicit filesystem path
- optional `cache_dir`, `revision`, and `trust_remote_code` overrides

Example cache-only configuration:

```yaml
model:
  pretrained_name: Qwen/Qwen3-0.6B
  source:
    mode: hub
    local_files_only: true
```

Example forced-local configuration:

```yaml
model:
  pretrained_name: /absolute/path/to/local-model
  source:
    mode: local
    path: /absolute/path/to/local-model
```

## Compression Output Boundary

The current compression workflow produces two different kinds of outputs:

- a loadable recovered model with updated weights in the original tensor shapes, typically under a `finetuned` directory
- a compressed export with variable-bit metadata and serialized weights, typically under a `compressed` directory

This split is intentional in 0.2.0.

The recovered model is the directly usable artifact for continued evaluation or inference with the original architecture shape.

The compressed export is currently the measurement artifact used to quantify compression results.

## Current Limitations

- Physical pruning is not a supported public 0.2.0 path. The codebase contains an incomplete physical-pruning branch, but the retained public configs use logical pruning with `physical: false`.
- Lottery Hypothesis style workflows are not part of the current public pipeline surface.
- The exported quantized artifact is not yet a standard Hugging Face `from_pretrained` target.
- MetaPAC provides a custom quantized loader in `metapac.src.compression.load_quantized_model`, but there is not yet a fully Hugging Face-compatible public loading contract for compressed exports.