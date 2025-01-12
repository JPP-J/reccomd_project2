import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tf_model(path):
    # Load and prepare data
    df = pd.read_csv(path)
    n_users = len(np.unique(df['user_id']))
    n_items = len(np.unique(df['book_id']))
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
    x = layers.concatenate([user_embedding, item_embedding])
    x = layers.Dense(128, activation='relu')(x)  # First hidden layer
    x = layers.Dropout(0.2)(x)  # Add Dropout after the first dense layer
    x = layers.Dense(64, activation='relu')(x)   # Second hidden layer
    x = layers.Dropout(0.2)(x)  # Add Dropout after the second dense layer

    # Output layer: Predict the rating (e.g., 1-5 scale)
    output = layers.Dense(1)(x)

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

