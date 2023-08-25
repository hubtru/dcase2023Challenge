from keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

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

    checkpoint_filepath = "checkpoint_convmixer/best_model_convmixer.h5"
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


def get_conv_mixer_256_8(
    image_size=256, filters=256, depth=8, kernel_size=8, patch_size=5, num_classes=1
):
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
    outputs = layers.Dense(num_classes, activation="softmax")(x)

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

    checkpoint_filepath = "checkpoint_convmixer_simple"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

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
    return dataset.prefetch(auto)
