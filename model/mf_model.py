import numpy as np
import torch
import torch.nn as nn
from base.base_model import BaseModel


class MatrixFactorization(BaseModel):
    """
    Matrix Factorization for Collaborative Filtering
    """
    def __init__(self, num_users, num_items, emb_dim=64, reg_lambda=0.01):
        """
        Initialize the Matrix Factorization model
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            emb_dim: Embedding dimension
            reg_lambda: Regularization parameter
        """
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.reg_lambda = reg_lambda
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize biases
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, users, items):
        """
        Forward pass of the model
        
        Args:
            users: User indices
            items: Item indices
            
        Returns:
            Predicted ratings
        """
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        # Element-wise product and sum
        dot = torch.sum(user_emb * item_emb, dim=1)
        
        # Add biases
        user_b = self.user_bias(users).squeeze()
        item_b = self.item_bias(items).squeeze()
        
        return dot + user_b + item_b + self.global_bias
    
    def regularization_loss(self):
        """
        Compute L2 regularization loss
        
        Returns:
            Regularization loss
        """
        reg_loss = torch.norm(self.user_embedding.weight, p=2) + torch.norm(self.item_embedding.weight, p=2)
        reg_loss += torch.norm(self.user_bias.weight, p=2) + torch.norm(self.item_bias.weight, p=2)
        return self.reg_lambda * reg_loss