# for test purposes
import pandas as pd
import torch
import numpy as np
from model import SalesNN, ImprovedSalesNN
from data_preparation import prepare_data, prepare_features, scale_features, get_feature_columns, create_sku_index

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict_sales(model, X, sku):
    with torch.no_grad():
        X = torch.FloatTensor(X)
        sku = torch.LongTensor(sku)
        predictions = model(X, sku)
    return predictions.numpy()

def get_predictions(test_data, model_path, scaler_path):
    # Prepare data
    test_data = prepare_data(test_data)
    test_data = prepare_features(test_data)

    feature_columns = get_feature_columns(test_data)

    test_data, sku_to_index = create_sku_index(test_data)

    X = test_data[feature_columns]
    sku = test_data['SKU_INDEX']
    
    X_scaled = scale_features(X, scaler_path)
    
    # Load model
    model = load_model(model_path)
    
    # Make predictions
    predictions = predict_sales(model, X_scaled, sku)
    
    # Add predictions to the test dataframe
    test_data['predictions'] = predictions
    
    return test_data


def test_get_prediction():
    test_data = pd.read_csv('data/test_data.csv')
    weekly_sales = get_predictions(test_data, 'data/NN_sales_prediction_model.pth', 'data/scaler.save')

print(test_get_prediction())