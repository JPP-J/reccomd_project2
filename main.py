import pandas as pd
import numpy as np
import pickle
import os

from utils.recomend_extended import *

# Load and prepare data
path = "https://drive.google.com/uc?id=1tRgURjUZ4OSKKhAUSpA5anWs_ZzOQ6Nw"
metadata_path = "https://drive.google.com/uc?id=15tQrIEUNzLOBeLNeDihK20IbAgzjejay"
item_metadata = pd.read_csv(metadata_path)  # Contains 'item_id' and 'description and eg.,'
print(item_metadata.tail())
print(item_metadata.columns)

# Run recommendation analysis
recommender, random_user = run_recommendation_analysis(
    ratings_path=path,
    user_col='userId',
    item_col='movieId',
    rating_col='rating',
    frac=1.00,
    test_size=0.2,
    n_neighbors=10,
    n_recommendations=10
)

# Add semantic capabilities
# # Generate query embedding using model from SentenceTransformer('all-MiniLM-L6-v2')
recommender.add_item_embeddings(item_metadata, item_col='movieId')

# Hybrid recommendation
query = "drama Thriller"
recommendations = recommender.hybrid_recommendation(random_user, query)
print(f"Recommendations for user {random_user} with query '{query}':", recommendations)

# # Saved model
# os.makedirs('model', exist_ok=True)
# with open('model/recommender.pkl', 'wb') as f:
#     pickle.dump(recommender, f)


