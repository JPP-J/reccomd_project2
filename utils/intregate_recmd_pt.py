import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

# PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define a simple model using Embeddings for Collaborative Filtering
class CFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)    # First hidden layer
        self.fc2 = nn.Linear(128, 64)        # Second hidden layer
        self.fc3 = nn.Linear(64, 1)          # Output layer

    def forward(self, user_ids, item_ids):
        # Debugging the max user_id and the number of embeddings
        # print(f"Max user ID: {user_ids.max().item()}, Num embeddings: {self.user_embedding.num_embeddings}")
        # print(f"Max user ID: {user_ids.max().item()}, Min user ID: {user_ids.min().item()}")
        #
        # print(f"user_ids range: {user_ids.min().item()} to {user_ids.max().item()}")
        # print(f"item_ids range: {item_ids.min().item()} to {item_ids.max().item()}")

        # Ensure indices are within valid range
        assert user_ids.max().item() < self.user_embedding.num_embeddings, "User ID out of range"
        assert user_ids.min().item() >= 0, "User ID contains negative values"
        assert item_ids.max().item() < self.item_embedding.num_embeddings, "Item ID out of range"
        assert item_ids.min().item() >= 0, "Item ID contains negative values"

        # Forward pass
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        x = torch.cat([user_embeds, item_embeds], dim=-1)   # Concatenate embeddings
        x = torch.relu(self.fc1(x))     # First hidden layer + ReLU
        x = self.dropout(x)             # Dropout after first hidden layer
        x = torch.relu(self.fc2(x))     # Second hidden layer + ReLU
        x = self.dropout(x)             # Dropout after second hidden layer
        x = self.fc3(x)                 # Output layer

        # Constrain predictions to the range [1, 5] = [a,b] : scaled_value=original_value×(b−a)+a
        x = torch.sigmoid(x) * 4 + 1  # Scale sigmoid output to [1, 5]
        return x

def CF_model_trian(path, frac=0.025, epoch=10, batch_size=16, embedding_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # Clear GPU cache

    # Load and prepare data
    print(f'Initial load data....')
    df = pd.read_csv(path)
    print(f'sized of df before : {df.shape}')
    df = df.sample(frac=frac, random_state=42)
    print(f'sized of df after: {df.shape}')

    # Reindex user and item IDs after sampling
    df['user_id'] = df['user_id'] - df['user_id'].min()  # Reindex user IDs
    df['book_id'] = df['book_id'] - df['book_id'].min()  # Reindex item IDs

    # Ensure the reindexing worked correctly
    assert df['user_id'].min() == 0, "User IDs are not zero-based"
    assert df['book_id'].min() == 0, "Item IDs are not zero-based"

    # Calculate the number of unique users and items
    n_users = df['user_id'].max() + 1  # Number of users is max user_id + 1
    n_items = df['book_id'].max() + 1  # Number of items is max item_id + 1
    print(f"n_users: {n_users}, n_items: {n_items}")

    # Ensure the maximum IDs are within the valid range
    assert df['user_id'].max() < n_users, "User IDs exceed the number of users"
    assert df['book_id'].max() < n_items, "Item IDs exceed the number of items"

    # Convert data to tensors
    user_ids = torch.tensor(df['user_id'].to_numpy(), dtype=torch.long).to(device)
    item_ids = torch.tensor(df['book_id'].to_numpy(), dtype=torch.long).to(device)
    ratings = torch.tensor(df['rating'].to_numpy(), dtype=torch.float32).to(device)

    # Create dataset and split into train/validation sets
    dataset = TensorDataset(user_ids, item_ids, ratings)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Reduced batch size
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)     # Reduced batch size

    # Create the model
    model = CFModel(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For mixed precision training

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f'Start training data....')
    epochs = epoch
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_user_ids, batch_item_ids, batch_ratings in train_dataloader:
            # print(f"Batch user IDs range: {batch_user_ids.min().item()} to {batch_user_ids.max().item()}")
            # print(f"Batch item IDs range: {batch_item_ids.min().item()} to {batch_item_ids.max().item()}")

            batch_user_ids = batch_user_ids.to(device)
            batch_item_ids = batch_item_ids.to(device)
            batch_ratings = batch_ratings.to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision
                outputs = model(batch_user_ids, batch_item_ids)
                loss = criterion(outputs.squeeze(), batch_ratings)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = outputs.squeeze().round()
            correct_train += (preds == batch_ratings).sum().item()
            total_train += batch_ratings.size(0)

        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = correct_train / total_train
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_user_ids, batch_item_ids, batch_ratings in val_dataloader:
                batch_user_ids = batch_user_ids.to(device)
                batch_item_ids = batch_item_ids.to(device)
                batch_ratings = batch_ratings.to(device)

                outputs = model(batch_user_ids, batch_item_ids)
                loss = criterion(outputs.squeeze(), batch_ratings)

                val_loss += loss.item()
                preds = outputs.squeeze().round()
                correct_val += (preds == batch_ratings).sum().item()
                total_val += batch_ratings.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        val_acc = correct_val / total_val
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print(f'Finished training data....')

    torch.save(model.state_dict(), "model/collaborative_filtering_param_model.pth")
    torch.save(model, "model/collaborative_filtering_model_complete.pth")
    torch.save(history, "model/training_history.pth")
    print("Model and training history saved!")


class saved_model_usage:
    def __init__(self, path_model, path_his):
        self.path_model = path_model
        self.path_his = path_his
        self.model = None           # Optional initialization
        self.history = None     # Optional initialization

    def load_model(self):
        # Load the saved model and training history
        self.model = torch.load(self.path_model)  # Load the model
        self.history = torch.load(self.path_his)  # Load the training history

        # Optionally load the model's state_dict if needed
        # model.load_state_dict(torch.load("collaborative_filtering_param_model.pth"))

        print(f'\nModel Architecture:\n{self.model}')

        return self.model, self.history

    def plot_saved_history(self):
        history = self.history

        # Plot the loss curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot the accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print the final metrics after plotting
        print("\nFinal Metrics:")
        print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")

def recommend_items(user_id, item_ids, model, interacted_items=None, top_n=5):
    """
    Generate top N recommendations for a user, excluding already interacted items.

    Args:
        user_id (int): The ID of the user.
        item_ids (list): List of all item IDs to consider.
        model: Trained PyTorch recommendation model.
        interacted_items (list): List of item IDs the user has already interacted with.
        top_n (int): Number of recommendations to return.

    Returns:
        list: Top N recommended item IDs and their scores.
    """
    # If no interacted items are provided, assume the user has interacted with none
    if interacted_items is None:
        interacted_items = []

    # Filter out invalid item IDs (outside the embedding layer's range)
    valid_item_ids = [item for item in item_ids if item < model.item_embedding.num_embeddings]

    # Exclude already interacted items
    candidate_items = [item for item in valid_item_ids if item not in interacted_items]

    # If no candidate items are left, return an empty list
    if not candidate_items:
        return []

    # Prepare input data
    user_input = torch.tensor([user_id] * len(candidate_items), dtype=torch.long)  # Repeat user ID for each candidate item
    item_input = torch.tensor(candidate_items, dtype=torch.long)  # List of candidate item IDs

    # Debugging: Print input data
    print(f"User input: {user_input}")
    print(f"Item input: {item_input}")

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    user_input = user_input.to(device)
    item_input = item_input.to(device)

    # Predict scores
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(user_input, item_input).squeeze().cpu().numpy()
        predictions = np.clip(predictions, 1, 5)  # Clip predictions to [1, 5]  # in case not re-trained

    # Combine item IDs with their predicted scores
    item_score_pairs = list(zip(candidate_items, predictions))

    # Sort by score in descending order
    item_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Return top N recommendations
    return item_score_pairs[:top_n]

def get_user_items(path, user_col:str, item_col:str):
    df = pd.read_csv(path)

    # Get a random user for demonstration
    random_user = np.random.choice(np.unique(df[user_col]))
    print(f"\nGenerating recommendations for user_id: {random_user}")

    # Filter the DataFrame for the specific user
    user_interactions = df[df[user_col] == random_user]

    # Extract the 'book_id' column from the filtered DataFrame
    interacted_items = user_interactions[item_col].tolist()
    print(f'\ninteracted_items: {interacted_items}')
    print(f'size of items: {len(interacted_items)}')

    # all items
    item_ids = np.unique(df[item_col])

    return  random_user, interacted_items, item_ids

