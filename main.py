import tensorflow as tf
import numpy as np
import time

from mfcc_preprocessing import compute_features, save_features, load_features_from_json, compute_all_features, load_all_features
from model_training import train_autoencoder, normalize_mfccs
from visualization import visualize_encoded_data, visualize_melspectrogram


# Define a custom callback for logging
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']}")


def main():
    start_time = time.time()
    # Main-Directory of datasets. Structure needs to be: main_directory\dev_fan\fan\train for the train dataset of the
    # subset fan.
    audio_all = r'C:\Users\Henning\Documents\Datasets_AD_Challenge'

    subsets = ['train', 'test']

    # On the current version the datasets ToyCar and ToyTrain are excluded, because the audio files are longer,
    # thus their np.arrays have a different structure

    # datasets = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    datasets = ['bearing', 'fan', 'gearbox', 'slider']

    # compute_all_features(audio_all, feature_type='mel', augment=False, num_augmentations=5, augmentation_factor=0.02,subsets=subsets, datasets=datasets)

    # Load the MFCC data from the JSON file
    data_train, data_test = load_all_features(feature_type="mfcc", subsets=subsets, datasets=datasets)
    visualize_melspectrogram(data_train[0])
    print(data_train.dtype)
    print(np.shape(data_train))
    print(data_test.dtype)
    print(np.shape(data_test))
    # autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

    # Normalize the data
    normalized_train_data = normalize_mfccs(data_train)
    normalized_test_data = normalize_mfccs(data_test)

    # Train the autoencoder
    encoding_dim = 32
    autoencoder = train_autoencoder(normalized_train_data, encoding_dim, epochs=20, batch_size=32,
                                    l2_reg=0.00, dropout_rate=0.4)

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
