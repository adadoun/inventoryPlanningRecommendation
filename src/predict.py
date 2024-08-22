# for test purposes
import pandas as pd
import torch
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error
from .model import SalesNN, ImprovedSalesNN
from .data_preparation import scale_features, get_feature_columns


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict_sales(model, X, sku):
    with torch.no_grad():
        X = torch.FloatTensor(X)
        sku = torch.LongTensor(np.array(sku))
        predictions = model(X, sku)
    return predictions.numpy()

def get_predictions(test_data, model_path, scaler_path):

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
