import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from utils.recomend_extended import *
from utils.intregate_recmd_tf import (tf_model, plot_history, saved_model_usage,
                                      recommend_items, get_user_items, tf_model_train)
from utils.intregate_recmd_pt import CFModel
from utils.sampling import get_stratified_sample, get_random_sample, create_batches, create_dataset, get_validation_sample
import os
import argparse



def train(path):
    print("Training the model...")

    # Load and prepare data
    tf_model_train(path, frac=0.025, epoch=10, batch_size=16, validation_split=0.2, embedding_dim=64)

def plot(path_model, path_his):
    print("Plotting results...")
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)

    print("\nLoading summary model...")
    model = loaded_model.load_model()

    print("\nLoading summary history...")
    history = loaded_model.load_history()

    print("\nLoading and plotting saved history...")
    loaded_model.plot_saved_history()

    return model, history

def test(path, path_model, path_his):
    print("Testing the model...")

    # Load the model directly
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)
    model = loaded_model.load_model()
    history = loaded_model.load_history()

    user_id, interacted_items, item_ids = get_user_items(path, user_col='user_id', item_col='book_id')

    # Generate recommendations
    recommendations = recommend_items(user_id, item_ids, model, interacted_items=interacted_items, top_n=5)

    # Display results
    print("Top Recommendations for User", user_id)
    for item_id, score in recommendations:
        print(f"Item ID: {item_id}, Predicted Score: {score:.4f}")


def main(path, path_model, path_his):
    parser = argparse.ArgumentParser(description="Script to train, plot, and test.")
    parser.add_argument('--mode', choices=['train', 'plot', 'test'], required=True,
                        help="Mode to run: train, plot, or test.")
    args = parser.parse_args()

    if args.mode == 'train':
        train(path)
    elif args.mode == 'plot':
        plot(path_model, path_his)
    elif args.mode == 'test':
        test(path, path_model, path_his)

if __name__ == '__main__':
    # Load data
    path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"
    path_model = "model/recommendation_model.keras"
    path_his = "model/training_history.pkl"

    # Set seed
    random.seed(42)  # Python random module
    np.random.seed(42)  # NumPy
    tf.random.set_seed(42)  # TensorFlow

    # Main Part
    main(path, path_model, path_his)


# Example usage in terminal : python reccmd_tf.py --mode plot
