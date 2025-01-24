import pandas as pd
import numpy as np
import pickle
import os

from utils.recomend_extended import *

# for training
def reccomd_sys(path, metadata_path):
    # Load and prepare data
    print(f'Loading data....\n')
    item_metadata = pd.read_csv(metadata_path)  # Contains 'item_id' and 'description and eg., genre'
    print(item_metadata.tail())
    print(item_metadata.columns)

    print(f'\nbuilding recommendation system....\n')
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
    print(f'\nbuilding recommendation system with semantic query....\n')

    # # Generate query embedding using model from SentenceTransformer('all-MiniLM-L6-v2')
    recommender.add_item_embeddings(item_metadata, item_col='movieId')

    # Hybrid recommendation
    query = "drama Thriller"
    recommendations = recommender.hybrid_recommendation(random_user, query)
    print(f"Recommendations for user {random_user} with query '{query}':", recommendations)

    return recommender

def usage_saved_model(metadata_path,
                      path_model: str = 'model/recommender.pkl',
                      query: str = None,
                      n_recommendations=10):
    print(f'Loading saved model....\n')

    with open(path_model, 'rb') as f:
        recommender = pickle.load(f)

    # Create a dictionary to map movieId to movie title
    item_metadata = pd.read_csv(metadata_path)
    movie_id_to_title = dict(zip(item_metadata['movieId'], item_metadata['title']))

    # Select a random user from the available users in the ratings dataset
    random_user = recommender.get_random_user()

    # Generate recommendations using all methods
    print(f'\nGernerate recommendations....\n')
    print(f"Recommendations for user {random_user}")

    rec_cf_user = recommender.recommend_by_user_cf(random_user, n_recommendations)
    rec_cf_item = recommender.recommend_by_item_cf(random_user, n_recommendations)
    rec_knn_user = recommender.recommend_by_user_knn(random_user, n_recommendations)
    rec_knn_item = recommender.recommend_by_item_knn(random_user, n_recommendations)
    rec_mf = recommender.recommend_by_mf(random_user, n_recommendations)

    print("\nUser-based CF recommendations:", rec_cf_user.index.tolist())
    print("Item-based CF recommendations:", rec_cf_item.index.tolist())
    print("User KNN recommendations:", rec_knn_user.index.tolist())
    print("Item KNN recommendations:", rec_knn_item.index.tolist())
    print("mf recommendations:", rec_mf.index.tolist())

    # Ensure query is provided
    if query is None:
        print("Error: Query parameter is required")
    else:
        # Get recommendations for the random user
        recommendations = recommender.hybrid_recommendation(random_user, query)

        # Convert movie IDs to movie titles
        recommended_titles = [movie_id_to_title.get(movie_id, "Unknown Movie") for movie_id in recommendations]
        print(f"\nRecommendations for user {random_user} with query '{query}': \n{recommended_titles}\n")

def save_model(recommender):
    # Saved model

    os.makedirs('model', exist_ok=True)
    with open('model/recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)

    print(f'\nsaving recommendation system complete!\n')


if __name__=='__main__':
    path = "https://drive.google.com/uc?id=1tRgURjUZ4OSKKhAUSpA5anWs_ZzOQ6Nw"
    metadata_path = "https://drive.google.com/uc?id=15tQrIEUNzLOBeLNeDihK20IbAgzjejay"

    # Training
    recommender = reccomd_sys(path, metadata_path)

    # Saved model to use later
    save_model(recommender)

    # Usage saved model
    usage_saved_model(metadata_path,
                      path_model= 'model/recommender.pkl',
                      query='Comedy',
                      n_recommendations=10)



