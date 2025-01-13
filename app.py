from flask import Flask, request, jsonify
import random
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained recommender system
with open('model/recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)

# Load item metadata (assuming it contains 'movieId' and 'title')
metadata_path = "https://drive.google.com/uc?id=15tQrIEUNzLOBeLNeDihK20IbAgzjejay"
item_metadata = pd.read_csv(metadata_path)  # Contains 'item_id' and 'description and eg.,'

# Create a dictionary to map movieId to movie title
movie_id_to_title = dict(zip(item_metadata['movieId'], item_metadata['title']))


@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the query parameter from the request
    query = request.args.get('query')

    # Ensure query is provided
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Select a random user from the available users in the ratings dataset
    random_user = recommender.get_random_user()

    # Get recommendations for the random user
    recommendations = recommender.hybrid_recommendation(random_user, query)

    # Convert movie IDs to movie titles
    recommended_titles = [movie_id_to_title.get(movie_id, "Unknown Movie") for movie_id in recommendations]

    # Return the recommendations as JSON
    return jsonify(recommended_titles)



if __name__ == '__main__':
    app.run(debug=True)
