import pandas as pd
import numpy as np
# from utils.recomend_extended import *
from utils.intregate_recmd_pt import CF_model_trian, CFModel, saved_model_usage, get_user_items, recommend_items
import torch
import os
import argparse

def train(path):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU")

    print("Training the model...")
    # Force CUDA operations to be synchronous
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # After the backward pass and optimizer step
    torch.cuda.empty_cache()

    # Training
    CF_model_trian(path=path, frac=0.05, epoch=10, batch_size=16, embedding_dim=64)

def plot(path_model, path_his):
    print("Plotting results...")
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)

    model, history = loaded_model.load_model()

    loaded_model.plot_saved_history()

    return  model, history

def test(path, path_model, path_his):
    print("Testing the model...")
    # Load the model directly
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)
    model, history = loaded_model.load_model()

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
    path_model = "model/collaborative_filtering_model_complete.pth"
    path_his = "model/training_history.pth"

    # Set seed
    np.random.seed(42)  # NumPy
    torch.manual_seed(42)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)  # PyTorch GPU
        torch.cuda.manual_seed_all(42)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensure deterministic CuDNN behavior
        torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility

    # Main Part
    main(path, path_model, path_his)


# example usage in terminal : python reccmd_pt.py --mode plot

