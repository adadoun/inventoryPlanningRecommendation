import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from .model import SalesNN


class SalesDataset(Dataset):
    """
    Custom PyTorch Dataset for sales data.

    This dataset holds the features (X), target values (y), and SKU identifiers
    for sales prediction tasks.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray or pd.Series): Target values.
        sku (np.ndarray or pd.Series): SKU identifiers.

    Attributes:
        X (torch.FloatTensor): Input features tensor.
        y (torch.FloatTensor): Target values tensor.
        sku (torch.LongTensor): SKU identifiers tensor.
    """

    def __init__(self, X, y, sku):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)
        self.sku = torch.LongTensor(sku.values)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, sku, target) for the requested sample.
        """
        return self.X[idx], self.sku[idx], self.y[idx]


def initialize_model(feature_columns, sku_train, device):
    """
    Initialize the SalesNN model, loss function, and optimizer.

    Args:
        feature_columns (list): List of feature column names.
        sku_train (pd.Series or np.array): Training data SKU identifiers.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        tuple: (model, criterion, optimizer)
    """
    # Initialize model
    salesnn_model = SalesNN(num_features=len(feature_columns),
                            num_skus=sku_train.nunique()).to(device)

    # Use L1Loss for MAE
    criterion = nn.L1Loss()

    # Initialize optimizer
    optimizer = optim.Adam(salesnn_model.parameters(), lr=0.00003)

    return salesnn_model, criterion, optimizer


def train_sales_nn(salesnn_model, train_loader, test_loader,
                   criterion, optimizer, num_epochs=20):
    """
    Train the Sales Neural Network model using Mean Absolute Error.

    Args:
        salesnn_model (nn.Module): The SalesNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        num_epochs (int, optional): Number of training epochs. Defaults to 20.

    Returns:
        tuple: Lists of training and test MAE for each epoch.
    """

    train_maes = []
    test_maes = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Training phase
        salesnn_model.train()
        train_mae = 0
        for X_batch, sku_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = salesnn_model(X_batch, sku_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_mae += loss.item()

        avg_train_mae = train_mae / len(train_loader)
        train_maes.append(avg_train_mae)

        # Evaluation phase
        salesnn_model.eval()
        test_mae = 0
        with torch.no_grad():
            for X_batch, sku_batch, y_batch in test_loader:
                y_pred = salesnn_model(X_batch, sku_batch)
                mae = criterion(y_pred, y_batch)
                test_mae += mae.item()

        avg_test_mae = test_mae / len(test_loader)
        test_maes.append(avg_test_mae)

        print(f"Epoch {epoch+1}/{num_epochs}, Train MAE: {avg_train_mae:.4f}, Test MAE: {avg_test_mae:.4f}")

    model_save_path = '../data/NN_sales_prediction_model.pth'

    # Save the entire model
    torch.save(salesnn_model, model_save_path)

    print(f"Model saved to {model_save_path}")

    return train_maes, test_maes
