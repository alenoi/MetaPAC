# src/metapac/evaluate_meta.py
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from metapac.models.mlp_regressor import MLPRegressor
from metapac.src.utils.analysis.metrics import mae, rmse, spearman_safe as spearman
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="meta_baseline_best.pt")
    parser.add_argument("--x_npz", type=str, required=True)  # contains X_test.npy, y_test.npy
    args = parser.parse_args()

    data = np.load(args.x_npz)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt["config"]
    # Create model with configuration from checkpoint
    model = MLPRegressor(
        in_dim=X_test.shape[1],
        hidden_sizes=[512, 256, 64],  # From checkpoint config
        dropout=0.15,  # From checkpoint config
        activation="relu",  # From checkpoint config
    ).eval()

    # Load pre-trained weights
    model.load_state_dict(ckpt["model"])

    ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    yp, yt = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            yp.append(pred.cpu())
            yt.append(yb.cpu())

    yp = torch.cat(yp)
    yt = torch.cat(yt)

    out = {"MAE": mae(yp, yt), "RMSE": rmse(yp, yt), "Spearman": spearman(yp, yt)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
