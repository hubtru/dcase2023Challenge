import tensorflow as tf
import numpy as np
import time

from preprocessing import compute_all_features, load_all_features
from model_training import train_autoencoder, normalize_features, train_autoencoder_conv
from visualization import visualize_encoded_data, visualize_melspectrogram, visualize_audio_length, visualize_datasets
from model_training import shuffle_data_and_labels


def main():
    start_time = time.time()
    # Main-Directory of datasets. Structure needs to be: main_directory\fan\train for the train dataset of the
    # subset fan.
    audio_all = r'C:\Users\HABELS.COMPUTACENTER\Downloads\dcase_training_data'
    datasets = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    # datasets = ['bearing']
    subsets = ['train', 'test']
    feature_options = ["mfcc", "mel", "stft", "fft"]

    feature = feature_options[2]
    output_size = (32, 256)

    # Visualizations before calculations
    # visualize_audio_length(audio_all)
    # visualize_datasets(audio_all)

    data_train, data_test, data_test_real_classification = compute_all_features(audio_all, datasets=datasets,
        feature_type=feature, augment=False, num_augmentations=5, augmentation_factor=0.02, output_size=output_size,
        save=False)

    # Load the features data from the JSON file
    # data_train, data_test = load_all_features(feature_type=feature, subsets=subsets,
                                          #    datasets=datasets, output_size=output_size)

    print(data_train.dtype)
    print(np.shape(data_train))
    print(data_test.dtype)
    print(np.shape(data_test))
    print(data_test_real_classification.dtype)
    print(np.shape(data_test_real_classification))
    print(data_test_real_classification[0])

    # Normalize the data
    normalized_train_data = normalize_features(data_train)
    normalized_test_data = normalize_features(data_test)
    shuffled_train = shuffle_data_and_labels(normalized_train_data)
    shuffled_test, shuffled_test_real_classification = shuffle_data_and_labels(normalized_test_data,
                                                                               data_test_real_classification)

    # Train the autoencoder
    encoding_dim = 128
    autoencoder = train_autoencoder_conv(shuffled_train, encoding_dim, epochs=10,
                                                            batch_size=16, l2_reg=0.002, dropout_rate=0.0)
    # Load Model
    # autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

    # Obtain the encoded representation of the input data
    print(np.shape(shuffled_train))
    print(np.shape(shuffled_test))
    encoded_train_data = autoencoder.predict(shuffled_train)
    encoded_test_data = autoencoder.predict(shuffled_test)

    # Calculate reconstruction errors for the encoded data
    train_reconstruction_errors = np.mean(np.square(shuffled_train - encoded_train_data), axis=(1, 2))
    test_reconstruction_errors = np.mean(np.square(shuffled_test - encoded_test_data), axis=(1, 2))

    # Classify anomalies/non-anomalies based on reconstruction errors
    threshold = np.percentile(train_reconstruction_errors, 70)
    train_predictions = train_reconstruction_errors > threshold
    test_predictions = test_reconstruction_errors > threshold

    # Visualizations after calculations
    visualize_encoded_data(encoded_train_data, encoded_test_data, train_predictions, test_predictions,
                           shuffled_test_real_classification)
    # visualize_melspectrogram(data_train[0])


    # Save the model
    # autoencoder.save('autoencoder_model.h5')

    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time} seconds")


# Define a custom callback for logging
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']}")


if __name__ == '__main__':
    main()
