import tensorflow as tf
import numpy as np


def normalize_mfccs(mfccs_data):
    # Calculate the mean and standard deviation along the axis of each feature
    mean = np.mean(mfccs_data, axis=(0, 2), keepdims=True)
    std = np.std(mfccs_data, axis=(0, 2), keepdims=True)

    # Perform the normalization
    normalized_data = (mfccs_data - mean) / std

    return normalized_data


def train_autoencoder(data, encoding_dim, epochs, batch_size):
    input_data = tf.keras.layers.Input(shape=(data.shape[1],))

    encoded = tf.keras.layers.Dense(64, activation='relu')(input_data)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(data.shape[1], activation='sigmoid')(decoded)

    # Define the autoencoder model
    autoencoder = tf.keras.Model(inputs=input_data, outputs=decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    for epoch in range(epochs):
        # Print the epoch number
        print(f"Epoch {epoch + 1}/{epochs}")

        # Fit the data to the autoencoder model
        autoencoder.fit(data, data, epochs=1, batch_size=batch_size, verbose=1)

    return autoencoder
