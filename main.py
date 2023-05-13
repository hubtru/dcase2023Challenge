import pandas as pd
import tensorflow as tf
import numpy as np

from mfcc_preprocessing import compute_mfccs, save_mfccs
import json
import numpy as np


def load_mfccs_from_json(json_file):
    with open(json_file, 'r') as f:
        mfccs_data = json.load(f)

    # Convert the MFCC data to a Pandas DataFrame
    df = pd.DataFrame.from_dict(mfccs_data).T
    return df


def main():
    audio_dir = r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\test'
    # Directory containing audio files
    output_file = 'mfccs_bearing_test.json'  # Output file to save MFCCs

    #mfccs_data = compute_mfccs(audio_dir)  # Compute MFCCs
    #save_mfccs(mfccs_data, output_file)  # Save MFCCs

    # Load the MFCC data from the JSON file
    bearing_train = load_mfccs_from_json(output_file)
    print(bearing_train.shape)

    # Assuming you have a pandas DataFrame called `df` with MFCCs data

    # Convert the DataFrame to a numpy array
    mfccs_data = bearing_train.values

    # Normalize the data
    normalized_data = (mfccs_data - np.mean(mfccs_data)) / np.std(mfccs_data)

    # Define the autoencoder model
    input_dim = mfccs_data.shape[1]
    encoding_dim = 32  # Adjust the encoding dimension as needed

    input_data = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = tf.keras.Model(inputs=input_data, outputs=decoded)

    # Compile and train the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(normalized_data, normalized_data, epochs=10, batch_size=32)

    # Obtain the encoded representation of the input data
    encoded_data = tf.keras.Model(inputs=input_data, outputs=encoded)
    encoded_output = encoded_data.predict(normalized_data)

    # Print the encoded representation
    print(encoded_output)
    print("this and")

    #print(bearing_train[0])


if __name__ == '__main__':
    main()
