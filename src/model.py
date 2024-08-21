import torch
import torch.nn as nn

class SalesNN(nn.Module):
    def __init__(self, num_features, num_skus, embedding_dim=32):
        super(SalesNN, self).__init__()
        self.sku_embedding = nn.Embedding(num_skus, embedding_dim)
        self.batch_norm1 = nn.BatchNorm1d(num_features + embedding_dim)
        self.fc1 = nn.Linear(num_features + embedding_dim, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, sku):
        sku_emb = self.sku_embedding(sku)
        x = torch.cat((x, sku_emb), dim=1)
        x = self.batch_norm1(x)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.batch_norm2(x)
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.batch_norm3(x)
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.batch_norm4(x)
        return self.fc4(x).squeeze()

class ImprovedSalesNN(nn.Module):
    def __init__(self, num_features, num_skus, embedding_dim=32):
        super(ImprovedSalesNN, self).__init__()
        self.sku_embedding = nn.Embedding(num_skus, embedding_dim)
        self.batch_norm1 = nn.BatchNorm1d(num_features + embedding_dim)
        self.fc1 = nn.Linear(num_features + embedding_dim, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, sku):
        sku_emb = self.sku_embedding(sku)
        x = torch.cat((x, sku_emb), dim=1)
        x = self.batch_norm1(x)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.batch_norm2(x)
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.batch_norm3(x)
        x = self.dropout(self.leaky_relu(self.fc3(x)))
        x = self.batch_norm4(x)
        return self.fc4(x).squeeze()