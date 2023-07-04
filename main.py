import tensorflow as tf
import numpy as np
import time
import keras_tuner as kt

from preprocessing import compute_all_features, load_all_features
from model_training import normalize_features, Autoencoder, ConvolutionalAutoencoder
from visualization import visualize_encoded_data, visualize_melspectrogram, visualize_audio_length, visualize_datasets
from model_training import shuffle_data_and_labels
from evaluation import calculate_harmonic_mean, calculate_auc, calculate_pauc, calculate_scores_for_machine_types, \
    compute_scores, prepare_input
from keras import losses


def main():
    start_time = time.time()
    audio_all = r'C:\Users\HABELS.COMPUTACENTER\Downloads\dcase_training_data'
    # datasets = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    datasets = ['valve']
    # datasets = ['bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
    # datasets = ['bandsaw']
    subsets = ['train', 'test']
    feature_options = ["mfcc", "mel", "stft", "fft"]
    model_options = ["Autoencoder", "ConvolutionalAutoencoder", None]
    load_model = True
    save_model = False
    epochs = 5
    l2reg = 0.002
    dropout_rate = 0

    model = model_options[1]
    feature = feature_options[2]
    output_size = (32, 256)
    model_name = f'saved_models/{model}_train_data_{feature}_epochs10_l2reg{l2reg}'

    # Visualizations before calculations
    # visualize_audio_length(audio_all)
    # visualize_datasets(audio_all)

    data_train, data_test, test_real_classification = compute_all_features(audio_all, datasets=datasets,
        feature_type=feature, augment=False, num_augmentations=5, augmentation_factor=0.02, output_size=output_size,
        save=False)

    # Load the features data from the JSON file
    # data_train, data_test = load_all_features(feature_type=feature, subsets=subsets,
                                          #    datasets=datasets, output_size=output_size)

    # Normalize the data
    normalized_train_data = normalize_features(data_train)
    normalized_test_data = normalize_features(data_test)
    # shuffled_train = shuffle_data_and_labels(normalized_train_data)
    normalized_test_data, test_real_classification = shuffle_data_and_labels(normalized_test_data,
                                                                             test_real_classification, 42)

    # Train the autoencoder
    encoding_dim = 128
    latent_dim = 64

    if load_model:
        autoencoder_class = tf.keras.models.load_model(model_name)
        autoencoder_class.fit(normalized_train_data, normalized_train_data,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(normalized_test_data, normalized_test_data))

    elif model == "Autoencoder":
        autoencoder_class = Autoencoder(input_shape=(normalized_train_data.shape[1],
                                                     normalized_train_data.shape[2]), latent_dim=latent_dim,
                                        l2_reg=l2reg, dropout_rate=dropout_rate)
        autoencoder_class.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder_class.fit(normalized_train_data, normalized_train_data,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(normalized_test_data, normalized_test_data))

    elif model == "ConvolutionalAutoencoder":
        autoencoder_class = ConvolutionalAutoencoder(input_shape=(normalized_train_data.shape[1],
                                                                  normalized_train_data.shape[2], 1),
                                                     latent_dim=latent_dim, l2_reg=l2reg, dropout_rate=dropout_rate)
        autoencoder_class.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder_class.fit(normalized_train_data, normalized_train_data,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(normalized_test_data, normalized_test_data))

    # autoencoder = train_autoencoder(shuffled_train, encoding_dim, epochs=5,
                                                           # batch_size=16, l2_reg=0.002, dropout_rate=0.0)

    # Obtain the encoded representation of the input data
    print(np.shape(normalized_train_data))
    print(np.shape(normalized_test_data))

    encoded_train_data = autoencoder_class.predict(normalized_train_data)
    encoded_test_data = autoencoder_class.predict(normalized_test_data)
    encoded_train_data = np.squeeze(encoded_train_data, axis=-1)
    encoded_test_data = np.squeeze(encoded_test_data, axis=-1)

    # Calculate reconstruction errors for the encoded data
    train_reconstruction_errors = np.mean(np.square(normalized_train_data - encoded_train_data), axis=(1, 2))
    test_reconstruction_errors = np.mean(np.square(normalized_test_data - encoded_test_data), axis=(1, 2))

    # Classify anomalies/non-anomalies based on reconstruction errors
    # threshold = np.percentile(train_reconstruction_errors, 95)
    threshold = 1
    train_predictions = train_reconstruction_errors > threshold
    test_predictions = test_reconstruction_errors > threshold

    # Visualizations after calculations
    visualize_encoded_data(encoded_train_data, encoded_test_data, train_predictions, test_predictions,
                            test_real_classification[:, 0])
    # visualize_melspectrogram(data_train[0])

    # Evaluation
    compute_scores(test_real_classification[:, 0], test_predictions)

    # Save the model
    if save_model:
        autoencoder_class.save(model_name)

    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time} seconds")


if __name__ == '__main__':
    main()
