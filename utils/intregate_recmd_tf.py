import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib


def tf_model(path):
    # Load and prepare data
    df = pd.read_csv(path)
    n_users = len(np.unique(df['user_id'])) + 1     # fix out of bound issue
    n_items = len(np.unique(df['book_id'])) + 1     # fix out of bound issue
    embedding_dim = 64  # Size of embedding vectors

    # User and item input layers
    user_input = layers.Input(shape=(1,), dtype=tf.int32, name="user_input")
    item_input = layers.Input(shape=(1,), dtype=tf.int32, name="item_input")

    # Embedding layers for users and items
    user_embedding = layers.Embedding(input_dim=n_users, output_dim=embedding_dim)(user_input)
    item_embedding = layers.Embedding(input_dim=n_items, output_dim=embedding_dim)(item_input)

    # Flatten the embedding vectors
    user_embedding = layers.Flatten()(user_embedding)
    item_embedding = layers.Flatten()(item_embedding)

    # Concatenate the embeddings and pass through neural network layers
    x = layers.concatenate([user_embedding, item_embedding])      # Concatenate embeddings
    x = layers.Dense(128, activation='relu')(x)             # First hidden layer
    x = layers.Dropout(0.2)(x)                                    # Add Dropout after the first dense layer
    x = layers.Dense(64, activation='relu')(x)              # Second hidden layer
    x = layers.Dropout(0.2)(x)                                    # Add Dropout after the second dense layer

    # Output layer: Predict the rating (e.g., 1-5 scale)
    # Output layer with sigmoid activation and scaling
    output = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid outputs in [0, 1]
    output = output * 4 + 1  # Scale to [1, 5]

    # Build the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model = models.Model(inputs=[user_input, item_input], outputs=output, callbacks=[early_stopping])

    # Compile the model with the Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    return df, model

def plot_history(history):
    history = history.history

    plt.figure(figsize=(12, 6))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

    # Print final metrics
    print("\nFinal Training Metrics from Saved History:")
    print(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Training Loss: {history['loss'][-1]:.4f}")
    print(f"Validation Loss: {history['val_loss'][-1]:.4f}")

class saved_model_usage:
    def __init__(self, path_model, path_his):
        self.path_model = path_model
        self.path_his = path_his
        self.model = None           # Optional initialization
        self.history_dict = None     # Optional initialization

    def load_model(self):  # to thrive data after trained
        # Load the model
        self.model = keras.models.load_model(self.path_model)

        # Recompile the model with metrics
        self.model.compile(
            optimizer='adam',  # or whatever optimizer you're using
            loss='mean_squared_error',  # or your loss function
            metrics=['accuracy']  # or your desired metrics
        )

        # model summary
        print(self.model.summary())

        return self.model

    def load_history(self):
        self.history_dict = joblib.load(self.path_his)
        return self.history_dict

    def plot_saved_history(self):
        # Load the saved history
        # saved_history = self.load_history(self.path_his)

        saved_history = self.history_dict.history

        # Create plots
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(saved_history['accuracy'], label='Training Accuracy')
        plt.plot(saved_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(saved_history['loss'], label='Training Loss')
        plt.plot(saved_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nFinal Training Metrics from Saved History:")
        print(f"Training Accuracy: {saved_history['accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {saved_history['val_accuracy'][-1]:.4f}")
        print(f"Training Loss: {saved_history['loss'][-1]:.4f}")
        print(f"Validation Loss: {saved_history['val_loss'][-1]:.4f}")


def recommend_items(user_id, item_ids, model, interacted_items=None, top_n=5):
    """
    Generate top N recommendations for a user, excluding already interacted items.

    Args:
        user_id (int): The ID of the user.
        item_ids (list): List of all item IDs to consider.
        model: Trained recommendation model.
        interacted_items (list): List of item IDs the user has already interacted with.
        top_n (int): Number of recommendations to return.

    Returns:
        list: Top N recommended item IDs and their scores.
    """
    # If no interacted items are provided, assume the user has interacted with none
    if interacted_items is None:
        interacted_items = []

    # Filter out invalid item IDs (outside the embedding layer's range)
    valid_item_ids = [item for item in item_ids if item < 10000]  # Adjust 10001 if needed

    # Exclude already interacted items
    candidate_items = [item for item in valid_item_ids if item not in interacted_items]

    # If no candidate items are left, return an empty list
    if not candidate_items:
        return []

    # Prepare input data
    user_input = np.array([user_id] * len(candidate_items))  # Repeat user ID for each candidate item
    item_input = np.array(candidate_items)  # List of candidate item IDs

    # Debugging: Print input data
    print(f"User input: {user_input}")
    print(f"Item input: {item_input}")

    # Predict scores
    predictions = model.predict([user_input, item_input]).flatten()
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