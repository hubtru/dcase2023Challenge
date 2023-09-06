import os

from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

from evaluation import compute_scores
from settings import start_model
from model_training import normalize_features, shuffle_data_and_labels
from preprocessing import compute_features, create_dataframe_from_filenames

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 10
image_size = 256
auto = tf.data.AUTOTUNE


data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"), ],
    name="data_augmentation",
)


def activation_block_autoencoder(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem_autoencoder(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block_autoencoder(x)


def conv_mixer_block_autoencoder(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block_autoencoder(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block_autoencoder(x)

    return x


def conv_mixer_block_autodecoder(x, filters: int, kernel_size: int):

    x0 = x
    x = layers.Tra(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block_autoencoder(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block_autoencoder(x)

    return x


def get_autoencoder_conv_mixer_256_8(input_shape=(256, 256, 1), filters=256, depth=8, kernel_size=5, patch_size=2,
                                     latent_dim=64):
    """Modified ConvMixer-based Autoencoder.
    """
    # Encoder
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem_autoencoder(x, filters, patch_size)

    # ConvMixer blocks (Encoder)
    encoder_outputs = []
    for _ in range(depth):
        x = conv_mixer_block_autoencoder(x, filters, kernel_size)
        encoder_outputs.append(x)

    # Latent space representation
    x = layers.GlobalAvgPool2D()(x)
    latent_space = layers.Dense(latent_dim)(x)

    # Decoder architecture (reverse the encoder)
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Reshape((1, 1, latent_dim))(decoder_inputs)

    for layer in reversed(encoder_outputs):
        x = conv_mixer_block_autoencoder(x, filters, kernel_size)
        x = layers.UpSampling2D(size=(2, 2))(x)  # Upsampling to match the original dimensions

    # Reconstruct the input
    decoded_output = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(x)

    # Full autoencoder model
    autoencoder_output = decoded_output
    autoencoder = keras.Model(inputs=[inputs, decoder_inputs], outputs=autoencoder_output)

    return autoencoder


def run_autoencoder_experiment(autoencoder, train_noisy_data, test_noisy_data):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",  # Use mean squared error as the reconstruction loss
    )

    checkpoint_filepath = "../../checkpoint_convmixer/best_model_convmixer.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="loss",  # Save based on training loss (reconstruction error)
        save_best_only=True,
        save_weights_only=True,
    )

    # Use the same data for input and target
    history = autoencoder.fit(
        [train_noisy_data, train_noisy_data],  # input and target are the same
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    autoencoder.load_weights(checkpoint_filepath)

    # Get the test reconstructions
    test_reconstructions = autoencoder.predict(test_noisy_data)
    # print(test_reconstructions.shape)
    # print(test_reconstructions[0])

    return history, autoencoder, test_reconstructions


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(image_size=96, filters=256, depth=8, kernel_size=8, patch_size=5, num_classes=1):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def run_experiment(model, train_dataset, test_dataset, val_dataset):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",  # sparse_categorical_crossentropy
        metrics=["accuracy"],
    )

    checkpoint_filepath = "cp_convmixer/supervised"

    # checkpoint_callback = tf.keras.callbacks.TensorBoard(checkpoint_filepath, histogram_freq=1)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",  # or "val_loss" depending on your preference
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    # model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(train_dataset)
    print(f"Train accuracy: {round(accuracy * 100, 2)}%")
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    _, accuracy = model.evaluate(val_dataset)
    print(f"Val accuracy: {round(accuracy * 100, 2)}%")

    return history, model


def visualization_plot(weights, idx=1):
    # First, apply min-max normalization to the
    # given weights to avoid isotrophic scaling.
    p_min, p_max = weights.min(), weights.max()
    weights = (weights - p_min) / (p_max - p_min)

    # Visualize all the filters.
    num_filters = 256
    plt.figure(figsize=(8, 8))

    for i in range(num_filters):
        # print(i)
        current_weight = weights[:, :, :, i]
        if current_weight.shape[-1] == 1:
            current_weight = current_weight.squeeze()
        ax = plt.subplot(16, 16, idx)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(current_weight)
        idx += 1


def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset  # dataset.prefetch(auto)


def main_convmixer_supervised():
    model = "ConvMixer"
    audio_all, datasets, output_size, feature = start_model()

    train_path = os.path.join(audio_all, datasets[0], "train")
    test_path = os.path.join(audio_all, datasets[0], "test")
    val_path = os.path.join(audio_all, datasets[0], "val")
    data_train, train_real_classification, train_filenames = compute_features(train_path, feature_type=feature,
                                                                             output_size=output_size)
    data_test, test_real_classification, test_filenames = compute_features(test_path, feature_type=feature,
                                                                             output_size=output_size)
    data_val, val_real_classification, val_filenames = compute_features(val_path, feature_type=feature,
                                                                             output_size=output_size)

    train_real_classification = create_dataframe_from_filenames(train_filenames)
    test_real_classification = create_dataframe_from_filenames(test_filenames)
    val_real_classification = create_dataframe_from_filenames(val_filenames)
    print(data_train[0])
    data_train = normalize_features(data_train, model)
    data_test = normalize_features(data_test, model)
    data_val = normalize_features(data_val, model)

    data_train, train_real_classification = shuffle_data_and_labels(data_train, train_real_classification)
    data_test, test_real_classification = shuffle_data_and_labels(data_test, test_real_classification)
    data_val, val_real_classification = shuffle_data_and_labels(data_val, val_real_classification)
    # data_val, val_real_classification = shuffle_data_and_labels(data_val, val_real_classification, 42)
    print(train_real_classification["anomaly"])
    print(test_real_classification["anomaly"])
    print(val_real_classification["anomaly"])

    data_train = data_train[:, :, :, np.newaxis]  # (x, 256, 256, 1)
    data_test = data_test[:, :, :, np.newaxis]
    data_val = data_val[:, :, :, np.newaxis]

    train_dataset = make_datasets(data_train, tf.convert_to_tensor(train_real_classification["anomaly"]), is_train=True)
    test_dataset = make_datasets(data_test, tf.convert_to_tensor(test_real_classification["anomaly"]))
    val_dataset = make_datasets(data_val, tf.convert_to_tensor(val_real_classification["anomaly"]))

    conv_mixer_model = get_conv_mixer_256_8(image_size=output_size[0])
    history, conv_mixer_model = run_experiment(conv_mixer_model, train_dataset, test_dataset, val_dataset)

    _, accuracy = conv_mixer_model.evaluate(data_train, train_real_classification["anomaly"])
    print(f"Train accuracy: {round(accuracy * 100, 2)}%")

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
    # compute_scores(test_real_classification["anomaly"], test_predictions)


if __name__ == '__main__':
    main_convmixer_supervised()
