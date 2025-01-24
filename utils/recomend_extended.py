import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from typing import Tuple, List, Dict, Set
from multiprocessing import Pool
from functools import partial
import os
import numba
from sentence_transformers import SentenceTransformer
import faiss
import random
import tf_keras as keras


class RecommenderSystem:
    def __init__(self, n_neighbors: int = 10,  n_factors: int = 10):
        """
        Initialize the recommender system with multiple approaches

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for KNN-based approaches
        n_factors : int
            Number of latent factors for Matrix Factorization
        """
        self.n_neighbors = n_neighbors
        self.n_factors = n_factors      # for Integrating Matrix Factorization
        self.user_item_matrix = None
        self.sparse_matrix = None
        self.item_similarity = None
        self.user_similarity = None
        self.user_knn = None
        self.item_knn = None
        self.ratings = None

        # Matrix Factorization components
        self.nmf_model = None
        self.user_factors = None
        self.item_factors = None

        # This Part for Integrate with LLM
        self.item_embeddings = None
        self.index = None
        self.item_metadata = None


    def fit(self, ratings_df: pd.DataFrame,
            user_col: str = 'user_id',
            item_col: str = 'item_id',
            rating_col: str = 'rating'):
        """
        Fit the recommender system with rating data
        """
        # store rating
        self.ratings = ratings_df

        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col
        ).fillna(0)

        # Create sparse matrix
        self.sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Compute similarity matrices
        self.user_similarity = cosine_similarity(self.sparse_matrix)
        self.item_similarity = cosine_similarity(self.sparse_matrix.T)

        # Fit KNN models
        self.user_knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.item_knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')

        self.user_knn.fit(self.sparse_matrix)
        self.item_knn.fit(self.sparse_matrix.T)

        # Fit Matrix Factorization model
        self.nmf_model = NMF(n_components=self.n_factors, init='random', random_state=42)
        self.user_factors = self.nmf_model.fit_transform(self.user_item_matrix.values)
        self.item_factors = self.nmf_model.components_

        return self

    def recommend_by_user_cf(self, user_id: int, n_recommendations: int = 10,
                             min_similarity: float = 0.0) -> pd.Series:
        """
        Generate recommendations using user-based collaborative filtering
        """

        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"User {user_id} not found in the dataset")

        # Get user's ratings and similarities
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity[user_idx]

        # Filter similar users
        similar_users = user_similarities > min_similarity
        if not any(similar_users):
            return pd.Series()

        # Calculate weighted ratings
        weighted_ratings = np.zeros(self.user_item_matrix.shape[1])
        similarity_sums = np.zeros(self.user_item_matrix.shape[1])

        for other_idx in range(len(self.user_item_matrix)):
            if other_idx != user_idx and similar_users[other_idx]: # need to be T AND T
                similarity = user_similarities[other_idx]
                ratings = self.user_item_matrix.iloc[other_idx].values

                weighted_ratings += similarity * ratings
                similarity_sums += np.abs(similarity) * (ratings != 0)

        # Avoid division by zero
        similarity_sums[similarity_sums == 0] = 1e-8
        recommendations = weighted_ratings / similarity_sums

        # Convert to series
        recommendations = pd.Series(
            recommendations,
            index=self.user_item_matrix.columns
        )

        # Filter out already rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        recommendations[user_ratings > 0] = -1

        return recommendations.nlargest(n_recommendations)

    def recommend_by_item_cf(self, user_id: int, n_recommendations: int = 10,
                             min_similarity: float = 0.0) -> pd.Series:
        """
        Generate recommendations using item-based collaborative filtering
        """
        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"User {user_id} not found in the dataset")

        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0] # if rating already

        if len(rated_items) == 0:
            return pd.Series()

        # Calculate weighted ratings
        weighted_sum = np.zeros(len(self.user_item_matrix.columns))
        similarity_sum = np.zeros(len(self.user_item_matrix.columns))

        for item_id, rating in rated_items.items():
            item_idx = self.user_item_matrix.columns.get_loc(item_id)
            similarities = self.item_similarity[item_idx]

            # Apply similarity threshold
            similarities[similarities < min_similarity] = 0

            weighted_sum += similarities * rating
            similarity_sum += np.abs(similarities)

        # Avoid division by zero
        similarity_sum[similarity_sum == 0] = 1e-8
        recommendations = weighted_sum / similarity_sum

        # Convert to series
        recommendations = pd.Series(
            recommendations,
            index=self.user_item_matrix.columns
        )


        # Filter out already rated items
        recommendations[user_ratings > 0] = -1

        return recommendations.nlargest(n_recommendations)

    def recommend_by_user_knn(self, user_id: int, n_recommendations: int = 10) -> pd.Series:
        """
        Generate recommendations using KNN-based user similarity
        """
        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"User {user_id} not found in the dataset")

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.sparse_matrix[user_idx:user_idx + 1] # cause 2D

        # Find nearest neighbors
        distances, indices = self.user_knn.kneighbors(user_vector)

        # Calculate weighted ratings
        weights = 1 / (distances.flatten() + 1e-8)
        weighted_sum = np.zeros(self.user_item_matrix.shape[1])
        weight_sum = np.zeros(self.user_item_matrix.shape[1])

        for idx, weight in zip(indices.flatten(), weights):
            weighted_sum += weight * self.user_item_matrix.iloc[idx].values
            weight_sum += weight  # like similarity

        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1e-8
        recommendations = weighted_sum / weight_sum

        # Convert to series
        recommendations = pd.Series(
            recommendations,
            index=self.user_item_matrix.columns
        )

        # Filter out already rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        recommendations[user_ratings > 0] = -1

        return recommendations.nlargest(n_recommendations)

    def recommend_by_item_knn(self, user_id: int, n_recommendations: int = 10) -> pd.Series:
        """
        Generate recommendations using KNN-based item similarity
        """
        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"User {user_id} not found in the dataset")

        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]

        if len(rated_items) == 0:
            return pd.Series()

        # Calculate recommendations for each rated item
        recommendations = np.zeros(len(self.user_item_matrix.columns))
        weights = np.zeros(len(self.user_item_matrix.columns))

        for item_id, rating in rated_items.items():
            item_idx = self.user_item_matrix.columns.get_loc(item_id)
            item_vector = self.sparse_matrix.T[item_idx:item_idx + 1]

            # Find nearest neighbors
            distances, indices = self.item_knn.kneighbors(item_vector)
            neighbor_weights = 1 / (distances.flatten() + 1e-8)

            for idx, weight in zip(indices.flatten(), neighbor_weights):
                recommendations[idx] += weight * rating
                weights[idx] += weight

        # Avoid division by zero
        weights[weights == 0] = 1e-8
        recommendations = recommendations / weights

        # Convert to series
        recommendations = pd.Series(
            recommendations,
            index=self.user_item_matrix.columns
        )

        # Filter out already rated items
        recommendations[user_ratings > 0] = -1

        return recommendations.nlargest(n_recommendations)

    def recommend_by_mf(self, user_id: int, n_recommendations: int = 10) -> pd.Series:
        """
        Generate recommendations using Matrix Factorization
        """
        if user_id not in self.user_item_matrix.index:
            raise KeyError(f"User {user_id} not found in the dataset")

        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Predict ratings using user and item factors
        predicted_ratings = np.dot(self.user_factors[user_idx], self.item_factors)

        # Convert to series
        recommendations = pd.Series(
            predicted_ratings,
            index=self.user_item_matrix.columns
        )

        # Filter out already rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        recommendations[user_ratings > 0] = -1

        return recommendations.nlargest(n_recommendations)


    def process_user(recommender_instance, user_id: int, test_data: pd.DataFrame, k_values: List[int]) -> Dict:
        """
        Process a single user for recommendation evaluation

        Parameters:
        - recommender_instance: The recommender system instance
        - user_id: User to evaluate
        - test_data: DataFrame with test interactions
        - k_values: List of k values for precision calculation

        Returns:
        - Dictionary of precision metrics for the user
        """
        # Store precision results for this user
        user_precisions = {}

        # Check if user exists in user item matrix
        if user_id not in recommender_instance.user_item_matrix.index:
            return user_precisions

        # Get actual items from test set for this user
        actual_items = set(test_data[test_data['user_id'] == user_id]['item_id'])

        # Skip if no actual items
        if len(actual_items) == 0:
            return user_precisions

        # Compute precisions for different k values and recommendation methods
        for k in k_values:
            try:
                # Generate recommendations for different methods
                rec_methods = {
                    'cf_user': set(recommender_instance.recommend_by_user_cf(user_id, k).index),
                    'cf_item': set(recommender_instance.recommend_by_item_cf(user_id, k).index),
                    'knn_user': set(recommender_instance.recommend_by_user_knn(user_id, k).index),
                    'knn_item': set(recommender_instance.recommend_by_item_knn(user_id, k).index)
                }

                # Calculate precision for each method
                for method_name, recommendations in rec_methods.items():
                    precision_key = f'precision@{k}_{method_name}'
                    user_precisions[precision_key] = len(recommendations & actual_items) / k

            except Exception as e:
                # Log or handle exceptions
                print(f"Error processing user {user_id}: {e}")
                continue

        return user_precisions

    def evaluate(self, test_data: pd.DataFrame, k_values: List[int] = [5, 10]) -> Dict:
        """
        Evaluate the recommender system using various metrics
        """
        if test_data.empty:
            raise ValueError("Test data is empty")

        if not all(col in test_data.columns for col in ['user_id', 'item_id']):
            raise ValueError("Test data missing required columns")

        metrics = {}

        for k in k_values:
            precision_cf_user = []
            precision_cf_item = []
            precision_knn_user = []
            precision_knn_item = []
            precision_mf = []

            for user_id in test_data['user_id'].unique():
                if user_id in self.user_item_matrix.index:
                    # Get actual items from test set
                    actual_items = set(test_data[test_data['user_id'] == user_id]['item_id'])

                    if len(actual_items) > 0:
                        # Get recommendations from each method
                        try:
                            rec_cf_user = set(self.recommend_by_user_cf(user_id, k).index)
                            rec_cf_item = set(self.recommend_by_item_cf(user_id, k).index)
                            rec_knn_user = set(self.recommend_by_user_knn(user_id, k).index)
                            rec_knn_item = set(self.recommend_by_item_knn(user_id, k).index)
                            rec_mf = set(self.recommend_by_mf(user_id, k).index)

                            # Calculate precision for each method
                            precision_cf_user.append(len(rec_cf_user & actual_items) / k)
                            precision_cf_item.append(len(rec_cf_item & actual_items) / k)
                            precision_knn_user.append(len(rec_knn_user & actual_items) / k)
                            precision_knn_item.append(len(rec_knn_item & actual_items) / k)
                            precision_mf.append(len(rec_mf & actual_items) / k)
                        except:
                            continue

            # Store metrics
            metrics[f'precision@{k}_cf_user'] = np.mean(precision_cf_user)
            metrics[f'precision@{k}_cf_item'] = np.mean(precision_cf_item)
            metrics[f'precision@{k}_knn_user'] = np.mean(precision_knn_user)
            metrics[f'precision@{k}_knn_item'] = np.mean(precision_knn_item)
            metrics[f'precision@{k}_mf'] = np.mean(precision_mf)

        return metrics


    def get_random_user(self):
        user_ids = self.ratings['user_id'].unique()  # Get unique user IDs from ratings data
        return random.choice(user_ids)  # Return a random user ID

    # -----------------------------------------------------------------------------------------------
    # Part Integrate with LLM
    def add_item_embeddings(self, item_metadata, item_col: str):
        """
        Generate embeddings for items using LLM and build FAISS index for semantic filtering.
        """
        item_metadata.rename(columns={item_col: 'item_id'}, inplace=True)
        # Combine title and genres into a single text column
        item_metadata['description'] = item_metadata['title'] + " " + item_metadata['genres']

        # Generate embeddings for the item descriptions (or titles + genres as needed)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        descriptions = item_metadata['description'].tolist()  # You can use 'title' or 'genres' if you prefer
        self.item_embeddings = model.encode(descriptions)

        # Build FAISS index for semantic filtering
        self.index = faiss.IndexFlatL2(self.item_embeddings.shape[1])
        self.index.add(self.item_embeddings)

        # Store the item metadata for future reference
        self.item_metadata = item_metadata

    def recommend_with_semantics(self, query, top_n=10):
        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')     # with 384-dimensional embeddings
        query_embedding = model.encode([query])

        # Perform semantic search using FAISS
        distances, indices = self.index.search(query_embedding, top_n)

        # Return the items corresponding to the indices
        return self.item_metadata.iloc[indices[0]]

    def hybrid_recommendation(self, user_id, query, cf_weight=0.7, llm_weight=0.3, top_n=10):
        # Get CF recommendations
        cf_recs = self.recommend_by_user_cf(user_id, top_n * 2)
        cf_scores = {item: rank for rank, item in enumerate(cf_recs.index, start=1)}

        # Get semantic recommendations
        semantic_recs = self.recommend_with_semantics(query, top_n * 2)
        semantic_scores = {item: rank for rank, item in enumerate(semantic_recs['item_id'], start=1)}

        # Combine and rank
        combined_scores = {}
        all_items = set(cf_scores.keys()).union(set(semantic_scores.keys()))

        for item in all_items:
            combined_scores[item] = cf_weight * (1 / (cf_scores.get(item, float('inf')))) + \
                                    llm_weight * (1 / (semantic_scores.get(item, float('inf'))))

        # Sort and return top recommendations
        ranked_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:top_n]]

# usage
def run_recommendation_analysis(ratings_path: str,
                                user_col: str,
                                item_col: str,
                                rating_col: str,
                                frac=0.005, test_size=0.2, n_neighbors=10, n_recommendations=10):
    """
    Run complete recommendation analysis
    """
    # Load and prepare data
    df_original = pd.read_csv(ratings_path)
    # df = df[df['user_id'] <= 1000]
    # Defines copy data from df_original to df
    df = pd.DataFrame()
    df['user_id'] = df_original[user_col]
    df['item_id'] = df_original[item_col]
    df['rating'] = df_original[rating_col]

    df_sample = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    train_data, test_data = train_test_split(df_sample, test_size=test_size, random_state=42)

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.values}")
    print(f"Sampling data shape: {df_sample.shape}")

    # Initialize and fit recommender
    recommender = RecommenderSystem(n_neighbors=n_neighbors)
    recommender.fit(train_data)

    # Get a random user for demonstration
    random_user = np.random.choice(recommender.user_item_matrix.index)
    # random_user = 7961
    print(f"\nGenerating recommendations for user_id: {random_user}")

    # Generate recommendations using all methods
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


    # Evaluate all methods
    metrics = recommender.evaluate(test_data, k_values=[5, 10])
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return recommender, random_user

# Usage example
if __name__ == "__main__":
    path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"
    recommender, random_user = run_recommendation_analysis(
        ratings_path=path,
        user_col='userId',
        item_col='movieId',
        rating_col='rating',
        frac=0.10,
        test_size=0.2,
        n_neighbors=10,
        n_recommendations=10
    )
