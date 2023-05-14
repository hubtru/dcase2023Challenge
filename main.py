import tensorflow as tf
import numpy as np

from mfcc_preprocessing import compute_mfccs, save_mfccs, load_mfccs_from_json, save_all_mfccs
from model_training import train_autoencoder, normalize_mfccs
from visualization import visualize_encoded_data


# Define a custom callback for logging
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']}")


def main():
    # Directory containing audio files
    audio_dir = [r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\train',
                 r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\test']

    # Output file to save MFCCs
    output_file = ['mfccs_bearing_train.json', 'mfccs_bearing_test.json',
                   'mfccs_bearing_train_augmented.json']

    # Load the MFCC data from the JSON file
    #bearing_train = compute_mfccs(audio_dir[0], True)
    #save_mfccs(bearing_train, "mfccs_bearing_train_augmented.json")
    bearing_train = load_mfccs_from_json(output_file[0])
    print(bearing_train.dtype)
    print(np.shape(bearing_train))

    # Normalize the data
    normalized_data = normalize_mfccs(bearing_train)

    # Reshape the input data to (None, input_dim)
    input_dim = bearing_train.shape[1] * bearing_train.shape[2]
    normalized_data = normalized_data.reshape(normalized_data.shape[0], -1)

    # Train the autoencoder
    encoding_dim = 32
    autoencoder = train_autoencoder(normalized_data, encoding_dim, epochs=10, batch_size=32)

    # Obtain the encoded representation of the input data
    encoded_data = autoencoder.predict(normalized_data)

    # Visualize the encoded data
    visualize_encoded_data(encoded_data)

    # Save the model
    autoencoder.save('autoencoder_model.h5')


if __name__ == '__main__':
    main()
