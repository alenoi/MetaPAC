# Dataset Handling

MetaPAC 0.2.0 is designed to work in online, cached, and fully local dataset workflows.

The active public configs currently cover three research datasets:

- SST-2, resolved internally as `glue/sst2`
- IMDB
- WOS, resolved internally as `waashk/wos11967`

The managed dataset repository can also cache local `DatasetDict` inputs for custom experiments and tests.

## Source Modes

Dataset loading is resolved through `metapac.src.utils.hf_sources.resolve_dataset_source`.

Supported source patterns:

- `mode: hub` loads a Hugging Face dataset reference, optionally with `local_files_only: true`
- `mode: disk` or `mode: local` loads a dataset previously saved with `datasets.load_from_disk`
- `mode: file` loads from explicit local data files through `datasets.load_dataset`

Typical hub-backed source:

```yaml
dataset:
  name: wos
  source:
    mode: hub
    local_files_only: true
    storage:
      root: metapac/artifacts/datasets
      mode: split
```

Typical manual local source:

```yaml
dataset:
  name: custom_local
  source:
    mode: disk
    path: /absolute/path/to/datasetdict
    storage:
      root: metapac/artifacts/datasets
      mode: raw
```

## Managed Storage

MetaPAC materializes datasets into a managed cache rooted by default at `metapac/artifacts/datasets`.

Two storage modes are used:

- `raw`: stores the original dataset snapshot without creating new validation or test splits
- `split`: stores a derived dataset with deterministic split processing applied

The cache path is fingerprinted from:

- dataset name and config
- source configuration
- split strategy and ratios
- deduplication and overlap rules
- random seed

This means repeated runs with the same config reuse the same materialized dataset instead of redownloading or rebuilding it.

## Split Strategies

MetaPAC supports two effective dataset handling patterns in the current code path.

Default handling:

- Uses an existing validation split if the dataset already has one
- Can create a validation split from `train` when `val_split_ratio` is set
- Keeps the dataset close to its original structure

Configured no-overlap handling:

- Uses `split_strategy: fixed_ratio_no_overlap`
- Rebuilds train, validation, and test splits from labeled text data
- Supports `deduplicate_by_text: true`
- Supports `enforce_no_text_overlap: true`

This is the pattern used by the offline Qwen WOS config.

## Offline-Oriented Workflows

MetaPAC treats offline usage as a first-class workflow rather than a special afterthought.

Supported patterns:

- Automatic download and reuse: use `mode: hub` with writable cache and storage root
- Cache-only hub execution: use `mode: hub` with `local_files_only: true`
- Fully manual local execution: use `mode: disk` and point to an existing dataset on disk

In practice this means:

- datasets can be fetched automatically from config when connectivity is available
- the same configs can be forced into cache-only behavior for disconnected runs
- manually prepared local datasets can be plugged into the same pipeline without changing the repository code

The managed repository also supports `allow_online_download: false` in storage configuration when a run must fail instead of retrying an online download.

## Current Scope Boundary

- The curated end-to-end public configs are written for SST-2, IMDB, and WOS.
- The dataset repository itself is more flexible and can cache custom local datasets.
- The automated tests validate raw and split caching behavior with synthetic local datasets, not with network downloads.