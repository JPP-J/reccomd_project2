import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


# Define a simple model using Embeddings for Collaborative Filtering
class CFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(CFModel, self).__init__()
        # Define embedding layers for users and items
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Define Dropout layers for regularization
        self.dropout = nn.Dropout(0.2)

        # Fully connected layers after the embeddings
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_ids, item_ids):
        # Get the embeddings for users and items
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        # Flatten embeddings
        x = torch.cat([user_embeds, item_embeds], dim=-1)

        # Pass through hidden layers with Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

def CF_model_trian(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare data
    df = pd.read_csv(path)

    n_users = len(np.unique(df['user_id']))
    n_items = len(np.unique(df['book_id']))

    df['user_id'] = df['user_id'] - df['user_id'].min()
    df['book_id'] = df['book_id'] - df['book_id'].min()

    user_ids = torch.tensor(df['user_id'].to_numpy(), dtype=torch.long).to(device)
    item_ids = torch.tensor(df['book_id'].to_numpy(), dtype=torch.long).to(device)
    ratings = torch.tensor(df['rating'].to_numpy(), dtype=torch.float32).to(device)

    dataset = TensorDataset(user_ids, item_ids, ratings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Create the model and move it to GPU
    model = CFModel(n_users=n_users, n_items=n_items, embedding_dim=64).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Iterate over the DataLoader batches
        for batch_user_ids, batch_item_ids, batch_ratings in dataloader:
            batch_user_ids = batch_user_ids.to(device)
            batch_item_ids = batch_item_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            # Forward pass
            outputs = model(batch_user_ids, batch_item_ids)

            # Compute the loss
            loss = criterion(outputs.squeeze(), batch_ratings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print progress
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "collaborative_filtering_model.pth")
    print("Model saved!")
