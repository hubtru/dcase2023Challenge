import tensorflow as tf
import numpy as np
import time
import keras_tuner as kt

from conv_mixer import get_conv_mixer_256_8, run_experiment, visualization_plot, \
    get_autoencoder_conv_mixer_256_8, run_autoencoder_experiment
from isolation_forest_detector import IsolationForestDetector
from preprocessing import compute_all_features, load_all_features
from model_training import normalize_features, Autoencoder, ConvolutionalAutoencoder
from visualization import visualize_encoded_data, visualize_melspectrogram, visualize_audio_length, visualize_datasets
from model_training import shuffle_data_and_labels
from evaluation import compute_scores

from keras import losses, layers
from keras_tuner import RandomSearch
from keras.models import Model
from keras.layers import Dense, Dropout, \
    Conv2D, MaxPooling2D, Input, UpSampling2D, Reshape, Flatten


class AutoencoderBlock(Model):
    def __init__(self, latent_dim, hp):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_layers = []
        for i in range(
            hp.Int("encoder_layers", min_value=2, max_value=5, step=1, default=0)
        ):
            self.encoder_layers.append(
                Dense(
                    units=hp.Choice("encoder_layers_{i}".format(i=i), [256]),
                    activation="relu",
                )
            )
        self.encoder_layers.append(Dense(latent_dim, activation="relu"))
        self.decoder_layers = []
        for i in range(
            hp.Int("decoder_layers", min_value=2, max_value=5, step=1, default=0)
        ):
            self.decoder_layers.append(
                Dense(
                    units=hp.Choice("decoder_layers_{i}".format(i=i), [256]),
                    activation="relu",
                )
            )
        self.decoder_layers.append(Dense(8192, activation="sigmoid"))

    def encode(self, encoder_input):
        encoder_output = Flatten()(encoder_input)
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        return encoder_output

    def decode(self, decoder_input):
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)
        decoder_output = Reshape((32, 256))(decoder_output)
        return decoder_output

    def call(self, x):
        return self.decode(self.encode(x))


def build_model(hp):
    latent_dim = 64
    autoencoder = AutoencoderBlock(latent_dim, hp)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def main():
    start_time = time.time()
    audio_all = r'C:\Users\HABELS.COMPUTACENTER\Downloads\dcase_training_data'
    # datasets = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
    # datasets = ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve', 'bandsaw', 'grinder',
               #  'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
    datasets = ['ToyTrain']
    # datasets = ['bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
    # datasets = ['bandsaw']
    feature_options = ["mfcc", "mel", "stft", "fft"]
    model_options = ["Autoencoder", "ConvolutionalAutoencoder", "IsolationForest", "ConvMixer", None]
    load_model = False
    save_model = False
    epochs = 10
    gaussion_noise = 0.0
    l2reg = 0.0001
    dropout_rate = 0

    model = model_options[3]
    feature = feature_options[2]
    output_size = (32, 96)
    model_name = f'saved_models/{model}_train_data_{feature}_epochs10_l2reg{l2reg}'
    checkpoint_path = f'checkpoints/best_model_{model}_newDense_{feature}_epochs{epochs}_l2reg{l2reg}_onlytrain.h5'

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
    data_train = normalize_features(data_train, model)
    data_test = normalize_features(data_test, model)

    # Add noise
    data_train_noise = data_train + gaussion_noise * tf.random.normal(shape=data_train.shape)
    print(data_train_noise.shape)
    data_test_noise = data_test + gaussion_noise * tf.random.normal(shape=data_test.shape)

    data_test, test_real_classification = shuffle_data_and_labels(data_test, test_real_classification, 42)
    '''
    tuner = RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=16,
        overwrite=True,
        directory="dense_256",
        project_name="dcase_fine_tune",
    )

    tuner.search(data_train, data_train, epochs=50, validation_data=(data_test, data_test))

    autoencoder = tuner.get_best_models(num_models=1)[0]
    tuner.results_summary(1)
    autoencoder.evaluate(data_test, data_test)
    '''
    # Train the autoencoder
    latent_dim = 64
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)

    if load_model:
        autoencoder_class = tf.keras.models.load_model(model_name)
        autoencoder_class.fit(data_train_noise, data_train,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(data_test_noise, data_test),
                              callbacks=[tensorboard_callback])

    elif model == "Autoencoder":
        autoencoder_class = Autoencoder(input_shape=(output_size[0] * output_size[1],), latent_dim=latent_dim,
                                        l2_reg=l2reg, dropout_rate=dropout_rate)
        autoencoder_class.compile(optimizer='adam', loss=losses.MeanSquaredError())
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                 monitor='val_loss',
                                                                 save_best_only=True,
                                                                 save_weights_only=True,
                                                                 mode='min',
                                                                 verbose=1)
        autoencoder_class.fit(data_train_noise, data_train,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(data_test_noise, data_test),
                              callbacks=[tensorboard_callback, checkpoint_callback])
        autoencoder_class.load_weights(checkpoint_path)

    elif model == "ConvolutionalAutoencoder":
        autoencoder_class = ConvolutionalAutoencoder(input_shape=(data_train.shape[1],
                                                                  data_train.shape[2], 1),
                                                     latent_dim=latent_dim, l2_reg=l2reg, dropout_rate=dropout_rate)
        autoencoder_class.compile(optimizer='adam', loss=losses.MeanSquaredError())
        autoencoder_class.fit(data_train_noise, data_train,
                              epochs=epochs,
                              shuffle=True,
                              validation_data=(data_test_noise, data_test),
                              callbacks=[tensorboard_callback])

    elif model == "IsolationForest":
        # Create and train Isolation Forest detector
        isolation_forest_detector = IsolationForestDetector(contamination=0.05)
        isolation_forest_detector.train(data_train)

        # Detect anomalies
        anomaly_scores, train_predictions = isolation_forest_detector.detect_anomalies(data_train_noise)
        anomaly_scores, test_predictions = isolation_forest_detector.detect_anomalies(data_test_noise)

    elif model == "ConvMixer":
        # conv_mixer_model, encoder, decoder = get_autoencoder_conv_mixer_256_8()
        # history, conv_mixer_model = run_autoencoder_experiment(conv_mixer_model, data_train_noise, data_test_noise)
        conv_mixer_model = get_conv_mixer_256_8()
        history, conv_mixer_model = run_experiment(conv_mixer_model, data_train_noise, data_test_noise)

        patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
        visualization_plot(patch_embeddings)

        for i, layer in enumerate(conv_mixer_model.layers):
            if isinstance(layer, layers.DepthwiseConv2D):
                if layer.get_config()["kernel_size"] == (5, 5):
                    print(i, layer)

        idx = 26  # Taking a kernel from the middle of the network.

        kernel = conv_mixer_model.layers[idx].get_weights()[0]
        kernel = np.expand_dims(kernel.squeeze(), axis=2)
        visualization_plot(kernel)

    # Obtain the encoded representation of the input data
    print(np.shape(data_train))
    print(np.shape(data_test))
    if model == "ConvolutionalAutoencoder" or model == "Autoencoder":
        encoded_train_data = autoencoder_class.predict(data_train_noise)
        encoded_test_data = autoencoder_class.predict(data_test_noise)
    else:
        encoded_train_data = data_train
        encoded_test_data = data_test

    if model == "ConvolutionalAutoencoder":
        encoded_train_data = np.squeeze(encoded_train_data, axis=-1)
        encoded_test_data = np.squeeze(encoded_test_data, axis=-1)
    elif model == "Autoencoder" or model == "IsolationForest":
        encoded_train_data = encoded_train_data.reshape(-1, output_size[0], output_size[1])
        encoded_test_data = encoded_test_data.reshape(-1, output_size[0], output_size[1])
        data_train = data_train.reshape(-1, output_size[0], output_size[1])
        data_test = data_test.reshape(-1, output_size[0], output_size[1])

    if model != "IsolationForest":
        # Calculate reconstruction errors for the encoded data
        train_reconstruction_errors = np.mean(np.square(data_train - encoded_train_data), axis=(1, 2))
        test_reconstruction_errors = np.mean(np.square(data_test - encoded_test_data), axis=(1, 2))

        # Calculate the mean and standard deviation of the training reconstruction errors
        mean_reconstruction_error = np.mean(train_reconstruction_errors)
        std_reconstruction_error = np.std(train_reconstruction_errors)

        # Set the threshold as two standard deviations above the mean
        threshold = mean_reconstruction_error + 2 * std_reconstruction_error

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
