"""Scikit-learn compatible wrapper for PyTorch models."""
import os
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TorchModelWrapper:
    """A scikit-learn compatible wrapper for PyTorch models.
    
    This wrapper makes PyTorch models compatible with scikit-learn's fit/predict
    interface, handling device management, batching, and data conversion.
    """

    def __init__(self, model: nn.Module, batch_size: int = 32, epochs: int = 100,
                 learning_rate: float = 0.001, device: str = None):
        """Initialize the wrapper.
        
        Args:
            model: PyTorch model to wrap
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        if self.device.type == "cpu":
            # Enable MKL optimizations for CPU
            torch.set_num_threads(os.cpu_count())
            torch.set_num_interop_threads(os.cpu_count())

        # Model setup
        self.model = model.to(self.device)
        print(f"Model architecture:\n{model}")

        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self
        """
        print("Starting model fitting...")
        # Standardize features
        X = self.scaler.fit_transform(X)

        try:
            # Convert to numpy if not already
            X = np.asarray(X)
            y = np.asarray(y)

            # Convert to torch tensors
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y.reshape(-1))  # Ensure 1D tensor

            # Optional validation tensors
            has_val = X_val is not None and y_val is not None
            if has_val:
                X_val = self.scaler.transform(np.asarray(X_val))
                y_val = np.asarray(y_val)
                X_val = torch.FloatTensor(X_val).to(self.device)
                y_val = torch.FloatTensor(y_val.reshape(-1)).to(self.device)

            # Create dataset and loader
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            print(f"Data loaded successfully. Shape: X={X.shape}, y={y.shape}")

            # Training loop
            self.model.train()
            self.training_history_ = []
            self.best_epoch_ = None
            self.best_val_rmse_ = None

            for epoch in range(self.epochs):
                t0 = perf_counter()
                total_loss = 0
                for batch_X, batch_y in loader:
                    # Move batch to device
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(batch_X)
                    loss = nn.MSELoss()(pred, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                cur_lr = float(self.optimizer.param_groups[0]["lr"])

                row = {
                    "epoch": int(epoch),
                    "train_mse": float(avg_loss),
                    "lr": cur_lr,
                    "elapsed_s": float(perf_counter() - t0),
                }

                if has_val:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(X_val)
                        val_mse = nn.MSELoss()(val_pred, y_val).item()
                        val_mae = nn.L1Loss()(val_pred, y_val).item()
                        val_rmse = float(np.sqrt(val_mse))
                    self.model.train()

                    row["val_mae"] = float(val_mae)
                    row["val_rmse"] = float(val_rmse)

                    if self.best_val_rmse_ is None or val_rmse < self.best_val_rmse_:
                        self.best_val_rmse_ = float(val_rmse)
                        self.best_epoch_ = int(epoch)

                self.training_history_.append(row)

                if (epoch + 1) % 10 == 0:
                    if has_val:
                        print(
                            f"Epoch {epoch + 1}/{self.epochs}, "
                            f"Loss: {avg_loss:.6f}, Val RMSE: {row['val_rmse']:.6f}"
                        )
                    else:
                        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

            print("Training completed successfully")

        except Exception as e:
            print(f"Error during training: {e}")
            raise

        return self

    def predict(self, X):
        """Generate predictions for input data.
        
        Args:
            X: Input features
            
        Returns:
            numpy array of predictions
        """
        try:
            # Standardize features
            X = self.scaler.transform(X)
            X = torch.FloatTensor(X)

            # Move to device in batches to prevent OOM
            self.model.eval()
            predictions = []

            with torch.no_grad():
                for i in range(0, len(X), self.batch_size):
                    batch_X = X[i:i + self.batch_size].to(self.device)
                    pred = self.model(batch_X)
                    predictions.append(pred.cpu())

            return torch.cat(predictions).numpy()

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
