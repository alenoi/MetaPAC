# MetaPAC: Meta-learning based Predictive Adaptive Compression

MetaPAC is a research codebase for predictive adaptive compression of transformer models. The repository supports a staged workflow in which a baseline model is fine-tuned, parameter-level hook statistics are converted into a meta-dataset, a meta-predictor is trained on those statistics, and the learned signal is then used to guide pruning and variable-bit quantization.

The active repository scope is the reproducible pipeline under `metapac/`, the curated entrypoint configs under `metapac/configs/`, the public documentation under `docs/`, and the automated tests under `tests/`. Large checkpoints, generated datasets, logs, and experiment outputs are intentionally excluded from version control.

## Features

- Meta-learning guided compression for ranking parameter importance before compression.
- Hybrid pruning and variable-bit quantization within a single staged pipeline.
- Modular compression architecture with explicit preparation, pruning, quantization, export, and validation phases.
- Offline-oriented model and dataset sourcing with managed raw and split dataset caches.
- Recovery fine-tuning support, including optional knowledge distillation.
- CLI-first workflow with a small public Python API surface for dataset building and predictor usage.

## Installation

For environment bootstrap and verification, see `docs/INSTALLATION.md`.

This branch is currently published as a source-first research repository rather than a packaged `pip install metapac` release.

Typical local setup:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Quick Start

### Using the CLI

Full automatic pipeline with a curated example config:

```bash
python -m metapac --config metapac/configs/auto_distilbert_sst2_fast.yaml
```

Run the default full pipeline without an explicit config path:

```bash
python -m metapac --mode auto
```

Run a single compression stage with the retained default compression config:

```bash
python -m metapac --mode compress --config metapac/configs/compress_distilbert_sst2.yaml
```

Build the meta-dataset only:

```bash
python -m metapac --mode feature_extract --config metapac/configs/feature_extraction.yaml
```

Train the default meta-predictor only:

```bash
python -m metapac --mode train_meta --config metapac/configs/meta_distilbert_sst2.yaml
```

### Using the Python API

The public Python surface retained in this branch includes:

- `build_meta_dataset`
- `BuildConfig`
- `TorchMetaPredictor`
- `TorchModelWrapper`
- `HookManager`
- `HookHFCallback`

## Pipeline Entry Points

MetaPAC exposes the following stages through `python -m metapac`:

- `baseline_finetune`
- `feature_extract`
- `train_meta`
- `compress`
- `auto`
- `auto:STAGE`

The default `auto` mode runs the full sequence `baseline_finetune -> feature_extract -> train_meta -> compress`.

The CLI accepts:

- `--mode MODE` to run a specific stage or `auto:STAGE` entrypoint.
- `--config PATH` to load a YAML config and optionally override the mode.

## Recommended Researcher Workflows

The curated auto configs currently intended for active use are:

- `metapac/configs/auto_distilbert_sst2_fast.yaml`
- `metapac/configs/auto_distilgpt2_imdb_fast.yaml`
- `metapac/configs/auto_qwen3_wos_fast.yaml`
- `metapac/configs/auto_qwen3_wos_fast_offline.yaml`

Example:

```bash
python -m metapac --config metapac/configs/auto_distilbert_sst2_fast.yaml
```

## Research Handoff Notes

- Baseline runs, meta-datasets, portable checkpoints, compressed models, and logs are generated locally and are gitignored by design.
- Model and dataset sources can be resolved from local paths or from cached Hugging Face assets, including `local_files_only` workflows for disconnected environments.
- Dataset materialization supports both raw caching and configured split caching under `metapac/artifacts/datasets`.
- Compression configs may use a meta checkpoint prefix such as `metapac/runs/checkpoints/metapac_meta_distilbert_sst2`; MetaPAC resolves this to the latest matching portable checkpoint directory.
- Direct compression entrypoints assume that the corresponding baseline run and meta-predictor checkpoint have already been produced.

## Runtime Layout

The active branch expects runtime-generated outputs to appear in these locations:

- `metapac/artifacts/raw` for hook statistics.
- `metapac/artifacts/datasets` for managed raw and split dataset caches.
- `metapac/artifacts/meta_dataset` for generated meta-datasets.
- `metapac/runs/checkpoints` for portable meta-predictor checkpoints.
- `targets/<model>/runs/...` for baseline fine-tuning outputs.
- `targets/<model>/models/experiments/.../finetuned` for loadable recovered models with updated weights in the original tensor shapes.
- `targets/<model>/models/experiments/.../compressed` for compressed exports and metadata used to measure compression results.

## Documentation Map

- `docs/INSTALLATION.md` - environment setup and verification
- `docs/DATASETS.md` - dataset sourcing, caching, splits, and offline workflows
- `docs/MODELS.md` - shared model handling, supported families, and current loader boundaries
- `docs/TESTING.md` - synthetic smoke test scope, commands, and limitations
- `metapac/configs/README.md` - curated config index

## Requirements

- Python 3.12
- PyTorch 2.x
- Transformers 4.44+
- Additional runtime and developer dependencies listed in `requirements.txt` and `requirements-dev.txt`

## Testing

The active automated suite is a synthetic smoke suite focused on pipeline behavior, artifact layout, and offline-safe execution. For scope, markers, and current boundaries, see `docs/TESTING.md`.

Run the active automated suite from the repository root:

```bash
python -m pytest tests -q
```

## Citation

If you use MetaPAC in research, please cite the metadata in `CITATION.cff`.

## License

This project is licensed under the MIT License. See `LICENSE` for the full text.