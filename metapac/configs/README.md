# MetaPAC Config Index

This directory contains the reduced public config surface for MetaPAC.

## Public Example Entrypoints

These configs are intended to remain as the main publishable examples:

- `auto_distilbert_sst2_fast.yaml` - Fast end-to-end encoder example for DistilBERT on SST-2. Runs baseline fine-tuning, hook collection, meta-dataset generation, meta-predictor training, then three-zone compression with logical pruning, 4-8 bit rank-aware quantization, recovery fine-tuning, and validation.
- `auto_distilgpt2_imdb_fast.yaml` - Fast end-to-end decoder-style example for DistilGPT2 on IMDB. Uses shorter sequences, GPT-style batching, and the same auto pipeline structure to show how the workflow maps to a causal language model classifier.
- `auto_qwen3_wos_fast.yaml` - Fast large-model example for Qwen3-0.6B on WOS. Uses fixed no-overlap dataset splits, gradient accumulation, larger sequence length, and a full compression run tuned for a heavier model.
- `auto_qwen3_wos_fast_offline.yaml` - Offline-capable variant of the Qwen3 WOS pipeline. Forces `local_files_only` model and dataset loading, stores cached datasets under `metapac/artifacts/datasets`, and enables explicit file logging for disconnected environments.

## Pipeline Defaults Kept For CLI Compatibility

These files are retained because the repository uses them as built-in defaults or direct documentation targets:

- `baseline_finetune.yaml` - Default Stage 0 config for fine-tuning a DistilBERT SST-2 baseline model. This is the file used when the CLI runs `baseline_finetune` mode without an override.
- `feature_extraction.yaml` - Default hook-to-meta-dataset config. Reads hook statistics from `metapac/artifacts/raw`, aggregates them into parameter-level features, and writes the meta-dataset outputs used by predictor training.
- `meta_distilbert_sst2.yaml` - Default meta-predictor training config for the DistilBERT SST-2 workflow. Defines feature inference, train/val/test splitting, the regression target, and the MLP training hyperparameters.
- `compress_distilbert_sst2.yaml` - Default full compression entrypoint for DistilBERT SST-2. Uses k-means zone assignment, variable-bit quantization, logical pruning, optional recovery fine-tuning, and validation against the latest matching meta checkpoint.
- `compress_with_finetuning.yaml` - Smaller example compression config that keeps pruning and quantization enabled while also enabling post-compression fine-tuning. Useful as a compact recovery example.
- `compress_with_pruning.yaml` - Smaller example compression config that keeps pruning and quantization enabled but leaves post-compression fine-tuning off by default. Useful for pruning-focused ablations.
- `strategy_defaults.yaml` - Shared fallback values for compression runs. The orchestrator loads this file when user configs omit compression, quantization, pruning, fine-tuning, or validation settings.
