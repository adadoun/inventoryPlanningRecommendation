import torch
import torch.nn as nn

class SalesNN(nn.Module):
    """
    Neural Network model for sales prediction.

    This model combines numerical features with SKU embeddings to predict sales.

    Args:
        num_features (int): Number of numerical input features.
        num_skus (int): Number of unique SKUs in the dataset.
        embedding_dim (int, optional): Dimension of the SKU embedding. Defaults to 32.

    Attributes:
        sku_embedding (nn.Embedding): Embedding layer for SKUs.
        fc1, fc2, fc3, fc4 (nn.Linear): Fully connected layers.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self, num_features, num_skus, embedding_dim=16):
        super(SalesNN, self).__init__()
        self.sku_embedding = nn.Embedding(num_skus, embedding_dim)
        self.fc1 = nn.Linear(num_features + embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, sku):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Tensor of numerical features.
            sku (torch.Tensor): Tensor of SKU indices.

        Returns:
            torch.Tensor: Predicted sales values.
        """
        sku_emb = self.sku_embedding(sku)
        x = torch.cat((x, sku_emb), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x).squeeze()
