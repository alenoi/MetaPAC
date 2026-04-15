# Testing

MetaPAC 0.2.0 currently ships an automated synthetic smoke suite under `tests/`.

The goal of this suite is to validate pipeline behavior, path layout, artifact emission, and core offline-safe execution without depending on large checkpoints, network downloads, or long-running training jobs.

## Run The Suite

Run the full suite from the repository root:

```bash
python -m pytest tests -q
```

Run a marker-specific subset:

```bash
python -m pytest tests -m preparation -q
python -m pytest tests -m pruning -q
python -m pytest tests -m quantization -q
python -m pytest tests -m pipeline -q
```

## Marker Layout

The active marker groups are defined in `pytest.ini`:

- `smoke`
- `preparation`
- `pruning`
- `quantization`
- `finetuning`
- `export`
- `pipeline`

## What The Tests Cover

The current suite validates:

- managed dataset repository behavior for raw and configured split caches
- baseline pipeline path layout and model handler routing
- preparation-stage hook-stat and feature extraction behavior
- logical pruning orchestration and metadata emission
- variable-bit quantization stage behavior
- post-compression fine-tuning stage wiring
- export and manifest generation
- end-to-end synthetic pipeline execution
- loader-facing compressed artifact behavior through dedicated smoke coverage

## How The Smoke Suite Works

The smoke tests deliberately avoid heavyweight runtime dependencies.

They use:

- tiny synthetic models and synthetic datasets
- temporary workspaces created per test
- stubbed or monkeypatched expensive stage operations where needed
- artifact-level assertions instead of long benchmark runs

The main helper surface lives in `tests/_smoke_helpers.py`.

## What The Tests Do Not Guarantee

The current automated suite does not claim to validate:

- training quality or accuracy on full real-world datasets
- performance regressions on large checkpoints
- online Hugging Face download behavior
- physical pruning as a supported release path
- Lottery Hypothesis workflows
- a fully Hugging Face-compatible loader for compressed exports

## Practical Interpretation

Passing tests mean that the active public pipeline structure is internally consistent and that the supported smoke-level code paths execute successfully.

Passing tests do not mean that every experimental path in the repository is production-ready or that unsupported compression outputs are deployable without additional tooling.