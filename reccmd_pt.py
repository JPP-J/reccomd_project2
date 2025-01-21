import pandas as pd
import numpy as np
# from utils.recomend_extended import *
from utils.intregate_recmd_pt import CF_model_trian, CFModel, saved_model_usage, get_user_items, recommend_items
import torch

# Load data
path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"

if __name__=='__main__':

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU")



# Training
if __name__ == '__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # After the backward pass and optimizer step
    torch.cuda.empty_cache()

    CF_model_trian(path=path, frac=0.05, epoch=10)

# After training
if __name__ == '__main__':
    path_model = r"C:\Users\topde\PycharmProjects\Projects\reccomd_project2\test\collaborative_filtering_model_complete.pth"
    path_model2 = r'C:\Users\topde\PycharmProjects\Projects\reccomd_project2\test\collaborative_filtering_param_model.pth'
    path_his = r"C:\Users\topde\PycharmProjects\Projects\reccomd_project2\test\training_history.pth"
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)

    model, history = loaded_model.load_model()

    loaded_model.plot_saved_history()

# reccomend for user
if __name__=="__main__":
    path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"
    user_id, interacted_items , item_ids = get_user_items(path, user_col='user_id', item_col='book_id')


    # Generate recommendations
    recommendations = recommend_items(user_id, item_ids, model, interacted_items=interacted_items, top_n=5)

    # Display results
    print("Top Recommendations for User", user_id)
    for item_id, score in recommendations:
        print(f"Item ID: {item_id}, Predicted Score: {score:.4f}")


