import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from .model import SalesNN
from .data_preparation import prepare_data, prepare_features, scale_features, get_feature_columns, create_sku_index

class SalesDataset(Dataset):
    def __init__(self, X, y, sku):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values)
        self.sku = torch.LongTensor(sku.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.sku[idx], self.y[idx]

def combined_loss(pred, target, alpha=0.5):
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return alpha * mse + (1 - alpha) * mae

def train_model(train_data, num_epochs=50, batch_size=128, learning_rate=0.000003, model_save_path='data/NN_sales_prediction_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    weekly_sales = prepare_data(train_data)
    weekly_sales = prepare_features(weekly_sales)
    weekly_sales, sku_to_index = create_sku_index(weekly_sales)
    
    feature_columns = get_feature_columns(weekly_sales)
    X = weekly_sales[feature_columns]
    y = weekly_sales['QUANTITY_SOLD']
    sku = weekly_sales['SKU_INDEX']
    
    X_scaled = scale_features(X)
    
    # Create datasets and dataloaders
    train_dataset = SalesDataset(X_scaled, y, sku)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = SalesNN(num_features=len(feature_columns), num_skus=len(sku_to_index)).to(device)
    criterion = combined_loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for X_batch, sku_batch, y_batch in train_loader:
            X_batch, sku_batch, y_batch = X_batch.to(device), sku_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch, sku_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss)
    
    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, sku_to_index
