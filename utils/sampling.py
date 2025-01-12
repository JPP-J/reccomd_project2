import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def get_random_sample(df, sample_size):
    return df.sample(n=sample_size, random_state=42)

def get_stratified_sample(df, test_size=0.2):
    train, _ = train_test_split(df, test_size=test_size, stratify=df['rating'])
    return train

def create_batches(df, batch_size=32):
    num_samples = len(df)
    batches = [df[i:i + batch_size] for i in range(0, num_samples, batch_size)]
    return batches

def create_dataset(df, batch_size=32, sample_size=10000):
    # Shuffle and sample the data
    df_sampled = df.sample(n=sample_size)

    # Convert the sampled dataframe into a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (df_sampled['user_id'].values, df_sampled['book_id'].values, df_sampled['rating'].values))

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def get_validation_sample(df, val_size=0.2):
    # Randomly sample for validation
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)
    return train_df, val_df





