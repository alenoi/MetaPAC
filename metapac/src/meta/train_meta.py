# src/meta/train_meta.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from metapac.src.utils.analysis.metrics import mae, rmse, spearman_safe, grouped_spearman
from ..data.meta_dataset import MetaDataset
from ..models.mlp_regressor import MLPRegressor
from ..utils.logger import TrainLogger, LogRow
from ..utils.seed import set_seed


# -----------------------------
# Loss: Huber (config alapján)
# -----------------------------
class HuberLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = input - target
        abs_err = torch.abs(err)
        delta = torch.tensor(self.delta, device=err.device, dtype=err.dtype)
        quad = torch.minimum(abs_err, delta)
        loss = 0.5 * quad ** 2 + delta * (abs_err - quad)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------
# Affín kalibráció (globális vagy group)
# --------------------------------------
@dataclass
class AffineCalib:
    a: float
    b: float


def _fit_affine(y_true: np.ndarray, y_pred: np.ndarray, ridge: float = 1e-6) -> Tuple[float, float]:
    X = np.stack([y_pred, np.ones_like(y_pred)], axis=1)
    XtX = X.T @ X + ridge * np.eye(2)
    Xty = X.T @ y_true
    a, b = np.linalg.solve(XtX, Xty)
    if a <= 0.0:
        a = max(1e-6, float(a))
    return float(a), float(b)


def fit_groupwise_affine(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        min_group_size: int = 100,
        ridge: float = 1e-6,
) -> Dict[str, AffineCalib]:
    out: Dict[str, AffineCalib] = {}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups).astype(str)

    a_glob, b_glob = _fit_affine(y_true, y_pred, ridge=ridge)
    out["__GLOBAL__"] = AffineCalib(a=a_glob, b=b_glob)

    for g in np.unique(groups):
        m = (groups == g)
        if m.sum() < min_group_size:
            out[g] = AffineCalib(a=a_glob, b=b_glob)
        else:
            a, b = _fit_affine(y_true[m], y_pred[m], ridge=ridge)
            out[g] = AffineCalib(a=a, b=b)
    return out


def apply_groupwise_affine(y_pred: np.ndarray, groups: np.ndarray, params: Dict[str, AffineCalib]) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.empty_like(y_pred)
    glob = params.get("__GLOBAL__")
    for i, g in enumerate(groups):
        p = params.get(g, glob)
        out[i] = p.a * y_pred[i] + p.b
    return out


# -----------------------------
# Config betöltése
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# Eval segédfüggvények
# -----------------------------
@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).squeeze(-1)
        preds.append(pred.detach().cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def evaluate(model, loader: DataLoader, device: torch.device, df_slice=None, split_name: str = "val") -> Dict[
    str, float]:
    y_pred, y_true = predict(model, loader, device)
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "spearman": spearman_safe(y_true, y_pred),
    }
    if df_slice is not None and "module" in df_slice.columns:
        groups = df_slice["module"].to_numpy()
        agg, per_group, skipped = grouped_spearman(y_true, y_pred, groups, weighted=True)
        out["spearman_group"] = agg
    return out, y_pred, y_true


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Validate that meta-dataset exists before starting training
    data_cfg = cfg.get("data", {})
    meta_dataset_path = Path(data_cfg.get("path", ""))

    if not meta_dataset_path.exists():
        print(f"[train_meta] ERROR: Meta-dataset not found at: {meta_dataset_path}")
        print("[train_meta] Please run feature extraction first to generate the meta-dataset:")
        print("  python -m metapac.src.pipeline --config metapac/configs/feature_extract.yaml --mode feature_extract")
        print("\nOr check that the 'data.path' in your config points to the correct location.")
        raise FileNotFoundError(f"Meta-dataset file not found: {meta_dataset_path}")

    # Check for columns.json manifest (optional but recommended)
    columns_json = meta_dataset_path.parent / "columns.json"
    if columns_json.exists():
        print(f"[train_meta] Found column manifest: {columns_json}")
    else:
        print(f"[train_meta] Warning: No columns.json found in {meta_dataset_path.parent}")
        print("[train_meta] Consider re-running feature extraction to generate column manifest.")

    print(f"[train_meta] Meta-dataset validated: {meta_dataset_path}")
    set_seed(int(cfg["training"]["seed"]))

    # Dataset
    dataset = MetaDataset(cfg["data"])
    X_tr, X_va, X_te, y_tr, y_va, y_te, df_tr, df_va, df_te = dataset.load()

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    # Dataloaderek
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    val_ds = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())
    test_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float())

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True,
                              num_workers=int(cfg["training"]["num_workers"]), pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                            num_workers=int(cfg["training"]["num_workers"]), pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
                             num_workers=int(cfg["training"]["num_workers"]), pin_memory=(device.type == "cuda"))

    # Modell
    model = MLPRegressor(
        in_dim=X_tr.shape[1],
        hidden_sizes=list(map(int, cfg["model"]["hidden_sizes"])) if "hidden_sizes" in cfg["model"] else [256, 128, 64],
        dropout=float(cfg["model"].get("dropout", 0.1)),
        activation=str(cfg["model"].get("activation", "relu")),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                              weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    # Loss
    use_huber = str(cfg["training"].get("loss", "huber")).lower() == "huber"
    huber_delta = float(cfg["training"].get("huber_delta", 1.0))
    criterion = HuberLoss(delta=huber_delta) if use_huber else torch.nn.MSELoss()

    # Logger
    run_name = cfg.get("experiment_name", "meta_baseline")
    logger = TrainLogger(run_dir="runs", run_name=run_name, use_progress=False)
    total_steps = len(train_loader)

    best_val_rmse = float("inf")
    patience = int(cfg["training"]["early_stop_patience"])
    stall = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(int(cfg["training"]["max_epochs"])):
        t0 = perf_counter()
        model.train()
        epoch_loss = 0.0

        prog = tqdm(train_loader, total=total_steps, leave=False, ncols=120, desc=f"Epoch {epoch:03d}")
        for xb, yb in prog:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * xb.size(0)
            cur_lr = optim.param_groups[0]["lr"]
            prog.set_postfix_str(f"loss={loss.item():.5f} lr={cur_lr:.2e}")

        epoch_loss /= max(1, len(train_loader.dataset))

        # Validation
        val_metrics, y_val_pred, y_val_true = evaluate(model, val_loader, device, df_slice=df_va, split_name="val")
        scheduler.step(val_metrics["rmse"])
        cur_lr = optim.param_groups[0]["lr"]

        improved = val_metrics["rmse"] < best_val_rmse - 1e-6
        if improved:
            best_val_rmse = val_metrics["rmse"]
            stall = 0
            # Save portable checkpoint with preprocessing pipeline
            checkpoint_dir = Path(f"artifacts/checkpoints/{run_name}_best")
            from metapac.src.models.meta_predictor import save_checkpoint_portable
            save_checkpoint_portable(
                checkpoint_dir=checkpoint_dir,
                model=model,
                imputer=dataset.imputer,
                scaler=dataset.scaler,
                feature_names=dataset.feature_names_,
                target_name=cfg["data"]["target"]["column_name"],
                task_type="regression",
                metadata={
                    "epoch": epoch,
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_spearman": val_metrics["spearman"],
                }
            )
        else:
            stall += 1

        elapsed = perf_counter() - t0
        logger.end_epoch(LogRow(epoch=epoch, step=total_steps, train_mse=epoch_loss, val_mae=val_metrics["mae"],
                                val_rmse=val_metrics["rmse"], val_spearman=val_metrics["spearman"], lr=cur_lr,
                                elapsed_s=elapsed, improved=improved))

        if stall >= patience:
            print("Early stopping.")
            break

    # -----------------------------
    # Load best és final eval
    # -----------------------------
    checkpoint_dir = Path(f"artifacts/checkpoints/{run_name}_best")
    model_state_path = checkpoint_dir / "model_state.pt"
    best = torch.load(model_state_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    # Uncalibrated test
    test_metrics, y_test_pred, y_test_true = evaluate(model, test_loader, device, df_slice=df_te, split_name="test")

    # -----------------------------
    # Post-hoc calibration (opcionális)
    # -----------------------------
    calib_enabled = bool(cfg["training"].get("calibration_enabled", False))
    calib_mode = str(cfg["training"].get("calibration_mode", "global"))

    if calib_enabled:
        group_col = cfg["data"]["split"]["group_col"]
        groups_val = df_va[group_col].astype(str).to_numpy()
        groups_test = df_te[group_col].astype(str).to_numpy()

        if calib_mode == "global":
            a, b = _fit_affine(y_val_true, y_val_pred, ridge=float(cfg["training"].get("calib_ridge", 1e-6)))
            calib_params = {"__GLOBAL__": AffineCalib(a=a, b=b)}
        elif calib_mode == "group":
            calib_params = fit_groupwise_affine(
                y_true=y_val_true,
                y_pred=y_val_pred,
                groups=groups_val,
                min_group_size=int(cfg["training"].get("calib_min_group_size", 100)),
                ridge=float(cfg["training"].get("calib_ridge", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown calibration_mode: {calib_mode}")

        y_test_pred_cal = apply_groupwise_affine(y_test_pred, groups_test, calib_params)

        test_mae_cal = mae(y_test_true, y_test_pred_cal)
        test_rmse_cal = rmse(y_test_true, y_test_pred_cal)
        test_spear_cal = spearman_safe(y_test_true, y_test_pred_cal)



    else:
        print("[calib] Calibration disabled (training.calibration_enabled=false)")
        test_mae_cal = test_metrics["mae"]
        test_rmse_cal = test_metrics["rmse"]
        test_spear_cal = test_metrics.get("spearman_group", test_metrics["spearman"])

    # Összegzés

    from metapac.src.utils.pretty_table import draw_table

    draw_table(
        headers=["", "MAE", "RMSE", "SPEARMAN"],
        rows=[
            ["uncalibrated", test_metrics["mae"], test_metrics["rmse"],
             test_metrics.get("spearman_group", test_metrics["spearman"])],
            ["calibrated", test_mae_cal, test_rmse_cal, test_spear_cal],
        ],
        # col_width=None  # ← nem adunk meg szélességet → AUTOFIT per oszlop
        padding=1,
        float_fmt=".3e",
        title="TEST metrics"
    )

    summary = {
        "val_best_rmse": float(best_val_rmse),
        "test_uncalib": {"mae": float(test_metrics["mae"]), "rmse": float(test_metrics["rmse"]),
                         "spearman": float(test_metrics.get("spearman_group", test_metrics["spearman"]))},
    }
    Path("metapac/runs").mkdir(parents=True, exist_ok=True)
    with open(f"metapac/runs/{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    print("Starting meta-model training")
    main()
