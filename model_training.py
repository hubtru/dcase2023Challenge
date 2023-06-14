import tensorflow as tf
import numpy as np


def normalize_mfccs(mfccs_data):

    if mfccs_data.shape[1] == 1:
        mfccs_data = np.squeeze(mfccs_data, axis=1)

    # Calculate the mean and standard deviation along the axis of each feature
    mean = np.mean(mfccs_data, axis=(0, 2), keepdims=True)
    std = np.std(mfccs_data, axis=(0, 2), keepdims=True)

    # Perform the normalization
    normalized_data = (mfccs_data - mean) / std

    print(np.shape(normalized_data))
    return normalized_data


def train_autoencoder(data, encoding_dim, epochs, batch_size, dropout_rate=0.0, l2_reg=0.00):
    input_data = tf.keras.layers.Input(shape=(data.shape[1], data.shape[2]))

    # Encoder
    encoded = tf.keras.layers.Dense(data.shape[2], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(input_data)
    encoded = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(encoded)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(encoded)

    # Dropout layer
    if dropout_rate > 0.0:
        dropout_encoded = tf.keras.layers.Dropout(dropout_rate)(encoded)
    else:
        dropout_encoded = encoded

    # Decoder
    decoded = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(dropout_encoded)
    decoded = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(decoded)
    decoded = tf.keras.layers.Dense(data.shape[2], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(decoded)

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
