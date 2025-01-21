import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
from utils.recomend_extended import *
from utils.intregate_recmd_tf import tf_model, plot_history, saved_model_usage, recommend_items, get_user_items
from utils.intregate_recmd_pt import CFModel
from utils.sampling import get_stratified_sample, get_random_sample, create_batches, create_dataset, get_validation_sample

# Load data
path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"

# Training part
if __name__ == "__main__":
    # Load and prepare data
    df, model = tf_model(path)

    # sampling df (e.g., random sampling 5% of the data)
    sampled_df = df.sample(frac=0.05, random_state=42)

    user_ids = sampled_df['user_id'].to_numpy()
    item_ids = sampled_df['book_id'].to_numpy()
    ratings =  sampled_df['rating'].to_numpy()

    user_ids = user_ids - user_ids.min()  # Adjust to start from 0
    item_ids = item_ids - item_ids.min()  # Adjust to start from 0

    print(f"Min user_id: {user_ids.min()}, Max user_id: {user_ids.max()}")
    print(f"Min item_id: {item_ids.min()}, Max item_id: {item_ids.max()}")

    # Train the model (using sample data)
    model.fit([user_ids, item_ids], ratings, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model to a file
    model.save('model/recommendation_model.keras')
    model.save('model/recommendation_model.h5') # optional
    print("Model saved as recommendation_model....")

    # Save the history to a file
    with open('model/training_history.pkl', 'wb') as f:
        pickle.dump(model.history, f)
    print("History saved as recommendation_model....")

    # Predict the rating for a specific user-item pair
    predicted_rating = model.predict([np.array([1]), np.array([101])])
    print(f"Predicted rating for user 1 and item 101: {predicted_rating[0][0]}")

    # Evaluate the model on the test set
    test_loss = model.evaluate([user_ids, item_ids], ratings)
    print(f"Test Loss: {test_loss}")


# Use after train
if __name__=='__main__':
    path_model = r"/model/recommendation_model.keras"
    path_his = r"/model/training_history.pkl"
    loaded_model = saved_model_usage(path_model=path_model, path_his=path_his)
    print("\nLoading summary model...")
    model = loaded_model.load_model()

    history = loaded_model.load_history()
    print("\nLoading and plotting saved history...")
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






