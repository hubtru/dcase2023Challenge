import pandas as pd
import tensorflow as tf
from mfcc_preprocessing import compute_mfccs, save_mfccs
import json
import numpy as np


def load_mfccs_from_json(json_file):
    with open(json_file, 'r') as f:
        mfccs_data = json.load(f)

    # Convert the MFCC data to a Pandas DataFrame
    df = np.array(mfccs_data)
    return df


def main():
    # Directory containing audio files
    audio_dir = [r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\train',
                 r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\test']

    # Output file to save MFCCs
    output_file = ['mfccs_bearing_train.json', 'mfccs_bearing_test.json']

    '''
    for i in range(len(audio_dir)):
        mfccs_data = compute_mfccs(audio_dir[i])  # Compute MFCCs
        save_mfccs(mfccs_data, output_file[i])  # Save MFCCs
    '''

    # Load the MFCC data from the JSON file
    bearing_train = load_mfccs_from_json(output_file[0])
    print(bearing_train.shape)

    # Convert the DataFrame to a numpy array
    print("hi")
    # Normalize the data
    '''mean = np.mean(mfccs_data, axis=0)
    print("hi")
    std = np.std(mfccs_data, axis=0)
    print("hi")
    normalized_data = (mfccs_data - mean) / std
    '''
    # Assuming `mfccs_data` is a 3D array

    mean = np.mean(bearing_train, axis=2)
    std = np.std(bearing_train, axis=2)
    normalized_data = (bearing_train - mean[:, :, np.newaxis]) / std[:, :, np.newaxis]

    #normalized_data = (mfccs_data - np.mean(mfccs_data)) / np.std(mfccs_data)
    print("hi")
    # Define the autoencoder model
    input_dim = mfccs_data.shape[1]
    encoding_dim = 32  # Adjust the encoding dimension as needed
    print("hi")
    input_data = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = tf.keras.Model(inputs=input_data, outputs=decoded)
    print("hi")
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print("hi")

    # Define a custom callback for logging
    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']}")

    # Train the autoencoder with logging
    autoencoder.fit(normalized_data, normalized_data, epochs=10, batch_size=32, callbacks=[LoggingCallback()])

    # Obtain the encoded representation of the input data
    encoded_data = tf.keras.Model(inputs=input_data, outputs=encoded)
    encoded_output = encoded_data.predict(normalized_data)

    # Print the encoded representation
    print(encoded_output)

    # Save the model
    autoencoder.save('autoencoder_model.h5')


if __name__ == '__main__':
    main()
