import torch
import numpy as np
import pandas as pd
from typing import List
from .model import SalesNN
from .data_preparation import scale_features, get_feature_columns


def load_model(model_path: str) -> SalesNN:
    """
    Load a trained PyTorch model from a file.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        SalesNN: The loaded PyTorch model in evaluation mode.
    """
    model = torch.load(model_path)
    model.eval()
    return model


def predict_sales(model: SalesNN, X: np.ndarray,
                  sku: np.ndarray) -> np.ndarray:
    """
    Make sales predictions using the provided model.

    Args:
        model (SalesNN): The trained sales prediction model.
        X (np.ndarray): The input features for prediction.
        sku (np.ndarray): The SKU indices corresponding to the input features.

    Returns:
        np.ndarray: The predicted sales values.
    """
    with torch.no_grad():
        X = torch.FloatTensor(X)
        sku = torch.LongTensor(np.array(sku))
        predictions = model(X, sku)
    return predictions.numpy()


def get_predictions(test_data: pd.DataFrame, model_path: str,
                    scaler_path: str) -> pd.DataFrame:
    """
    Generate sales predictions for the given test data.

    Args:
        test_data (pd.DataFrame): The test dataset containing features and SKU information.
        model_path (str): The path to the saved model file.
        scaler_path (str): The path to the saved scaler object.

    Returns:
        pd.DataFrame: The test dataset with an additional 'predictions' column containing the sales predictions.
    """
    feature_columns = get_feature_columns(test_data)
    X = test_data[feature_columns]
    sku = test_data["SKU_INDEX"]
    X_scaled = scale_features(X, scaler_path)

    # Load model
    model = load_model(model_path)

    # Make predictions
    predictions = predict_sales(model, X_scaled, sku)

    # Add predictions to the test dataframe
    test_data['predictions'] = predictions

    return test_data
