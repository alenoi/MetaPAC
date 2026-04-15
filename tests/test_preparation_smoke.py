from __future__ import annotations

import shutil

import pytest

from metapac.src.compression.phases.preparation import (
    compute_importance_scores,
    create_preprocessed_pipeline,
    extract_parameter_features,
    load_meta_predictor_checkpoint,
    rank_and_partition_parameters,
)


@pytest.mark.smoke
@pytest.mark.preparation
def test_preparation_builds_ranked_compression_plan(smoke_workspace) -> None:
    model, imputer, scaler, feature_names, target_name, task_type = load_meta_predictor_checkpoint(
        str(smoke_workspace.meta_checkpoint_dir)
    )
    pipeline = create_preprocessed_pipeline(model, imputer, scaler)

    parameter_features = extract_parameter_features(
        str(smoke_workspace.target_model_dir),
        feature_names,
    )
    importance_df = compute_importance_scores(pipeline, parameter_features, feature_names)
    ranked_df = rank_and_partition_parameters(
        importance_df,
        zones_config=smoke_workspace.build_config()["compression"]["zones"],
        zone_assignment_cfg={"method": "quantile"},
    )

    assert target_name == "importance"
    assert task_type == "regression"
    assert set(parameter_features["parameter_name"]) == set(smoke_workspace.parameter_names)
    assert parameter_features["parameter_name"].value_counts().max() > 1
    assert len(importance_df) == len(smoke_workspace.parameter_names)
    assert set(ranked_df["action"]) == {"keep", "prune", "quantize"}
    assert ranked_df["importance_score"].is_monotonic_decreasing


@pytest.mark.smoke
@pytest.mark.preparation
def test_preparation_uses_parent_run_hook_stats_for_selected_checkpoint(smoke_workspace) -> None:
    checkpoint_dir = smoke_workspace.target_model_dir / "checkpoint-7"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = smoke_workspace.target_model_dir / "artifacts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        smoke_workspace.target_model_dir / "runs" / "parameter_stats_epoch0.csv",
        raw_dir / "hook_stats_epoch0.csv",
    )

    _, _, _, feature_names, _, _ = load_meta_predictor_checkpoint(
        str(smoke_workspace.meta_checkpoint_dir)
    )

    parameter_features = extract_parameter_features(
        str(checkpoint_dir),
        feature_names,
    )

    assert set(parameter_features["parameter_name"]) == set(smoke_workspace.parameter_names)