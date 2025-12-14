"""Scikit-learn compatible wrapper for PyTorch models."""
import os

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

    def fit(self, X, y):
        """Fit the model to the data.
        
        Args:
            X: Input features
            y: Target values
            
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

            # Create dataset and loader
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            print(f"Data loaded successfully. Shape: X={X.shape}, y={y.shape}")

            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
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

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(loader)
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
