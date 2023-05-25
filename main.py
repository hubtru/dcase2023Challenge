import tensorflow as tf
import numpy as np
import time

from mfcc_preprocessing import compute_features, save_features, load_features_from_json, save_all_features
from model_training import train_autoencoder, normalize_mfccs
from visualization import visualize_encoded_data


# Define a custom callback for logging
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']}")


def main():
    start_time = time.time()
    # Directory containing audio files
    audio_dir = [r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\train',
                 r'C:\Users\Henning\Documents\Datasets_AD_Challenge\dev_bearing\bearing\test']

    # Output file to save MFCCs
    output_file = ['mfccs_bearing_train.json', 'mfccs_bearing_test.json',
                   'mfccs_bearing_train_augmented.json', 'mel_bearing_train.json',
                   'mel_bearing_test.json', 'mel_bearing_train_augmented.json',
                   'stft_bearing_train.json',
                   'stft_bearing_test.json', 'stft_bearing_train_augmented.json'
                   ]

    # Load the MFCC data from the JSON file

    # bearing_train = compute_features(audio_dir[1], feature_type='stft', augment=False, num_augmentations=5, augmentation_factor=0.02)

    # save_features(bearing_train, "stft_bearing_test.json")

    bearing_train = load_features_from_json(output_file[6])
    bearing_test = load_features_from_json(output_file[7])
    print(bearing_train.dtype)
    print(np.shape(bearing_train))
    print(bearing_test.dtype)
    print(np.shape(bearing_test))
    # autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

    # Normalize the data
    normalized_train_data = normalize_mfccs(bearing_train)
    normalized_test_data = normalize_mfccs(bearing_test)

    # Train the autoencoder
    encoding_dim = 32
    autoencoder = train_autoencoder(normalized_train_data, encoding_dim, epochs=10, batch_size=32,
                                    l2_reg=0.01, dropout_rate=0.0)

    # Obtain the encoded representation of the input data
    print(np.shape(normalized_train_data))
    print(np.shape(normalized_test_data))
    encoded_train_data = autoencoder.predict(normalized_train_data)
    encoded_test_data = autoencoder.predict(normalized_test_data)

    # Visualize the encoded data
    visualize_encoded_data(encoded_train_data, encoded_test_data)

    # Save the model
    # autoencoder.save('autoencoder_model.h5')

    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time} seconds")


if __name__ == '__main__':
    main()
