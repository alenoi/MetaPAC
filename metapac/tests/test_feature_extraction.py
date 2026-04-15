import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import json
import pandas as pd
from metapac.src.feature_extraction import build_meta_dataset, BuildConfig


def _make_dummy_csv(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    rows = []
    for epoch in [0, 1]:
        for step in range(2):
            rows.append({
                "run_id": "runA",
                "epoch": epoch,
                "step": step,
                "module": "encoder.layer.0",
                "phase": "train",
                "activation_values": json.dumps([[0.1, 0.2, 0.0], [0.3, 0.0, 0.5]]),
                "grad_values": json.dumps([[0.01, 0.0, 0.02], [0.0, 0.03, 0.0]]),
            })
    pd.DataFrame(rows[:2]).to_csv(os.path.join(tmpdir, "hook_stats_epoch0.csv"), index=False)
    pd.DataFrame(rows[2:]).to_csv(os.path.join(tmpdir, "hook_stats_epoch1.csv"), index=False)


def test_build_pipeline(tmp_path):
    src = tmp_path / "artifacts_src"
    out = tmp_path / "meta_dataset"
    _make_dummy_csv(str(src))
    cfg = BuildConfig(reducer="mean_pool", token_average=True, write_parquet=False, write_csv=False)
    path = build_meta_dataset(str(src), str(out), cfg)
    assert os.path.isdir(out)
    assert path.endswith('"metapac/artifacts/meta_dataset/meta_dataset.parquet"')
