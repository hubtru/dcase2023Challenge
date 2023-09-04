# from tensorflow.keras import layers
import os

from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

from model_training import normalize_features, shuffle_data_and_labels, normalize_datasets_to_01
from preprocessing import compute_features, create_dataframe_from_filenames
from settings import start_model
from visualization import plot_training_history

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 8
num_epochs = 100
filters = 256
depth = 3
kernel_size = 10
patch_size = 8
num_classes = 2
image_size = 96
directory = "experiments/convmixer/no_augmentation"

'''
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]


print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")
'''
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
'''
data_train = normalize_features(data_train, model)
data_test = normalize_features(data_test, model)
data_val = normalize_features(data_val, model)
'''
data_train, data_test, data_val = normalize_datasets_to_01(data_train, data_test, data_val)
data_train, train_real_classification = shuffle_data_and_labels(data_train, train_real_classification)
data_test, test_real_classification = shuffle_data_and_labels(data_test, test_real_classification)
data_val, val_real_classification = shuffle_data_and_labels(data_val, val_real_classification)
# data_val, val_real_classification = shuffle_data_and_labels(data_val, val_real_classification, 42)

data_train = data_train[:, :, :, np.newaxis]  # (x, 256, 256, 1)
data_test = data_test[:, :, :, np.newaxis]
data_val = data_val[:, :, :, np.newaxis]
auto = tf.data.AUTOTUNE
'''
data_augmentation = keras.Sequential(
    [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
    name="data_augmentation",
)
'''

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    '''if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    '''
    return dataset.prefetch(auto)


train_dataset = make_datasets(data_train, tf.convert_to_tensor(train_real_classification["anomaly"]), is_train=True)
test_dataset = make_datasets(data_test, tf.convert_to_tensor(test_real_classification["anomaly"]))
val_dataset = make_datasets(data_val, tf.convert_to_tensor(val_real_classification["anomaly"]))

# train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
# val_dataset = make_datasets(x_val, y_val)
# test_dataset = make_datasets(x_test, y_test)


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
    image_size=32, filters=256, depth=8, kernel_size=8, patch_size=5, num_classes=10):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 1))
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def run_experiment(model):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # binary
        metrics=["accuracy"],
    )

    checkpoint_filepath = "cp_convmixer/copied"
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
    '''
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(train_dataset)
    print(f"Train accuracy: {round(accuracy * 100, 2)}%")
    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    _, accuracy = model.evaluate(val_dataset)
    print(f"Val accuracy: {round(accuracy * 100, 2)}%")
    '''
    return history, model


def save_experiment_results(filters, depth, kernel_size, patch_size, num_classes, model):
    # Define the file name based on your settings
    # Define the directory path

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    file_name = f"ConvMixer_img{image_size}_depth{depth}_kernel{kernel_size}_patches{patch_size}.txt"
    # Define the file path
    file_path = os.path.join(directory, file_name)

    # Open the file for writing
    with open(file_path, 'w') as file:
        # Save history
        file.write("History:\n")
        file.write(str(history.history) + "\n")

        # Save accuracies
        _, accuracy = model.evaluate(train_dataset)
        file.write("Train accuracy: {:.2f}%\n".format(round(accuracy * 100, 2)))
        _, accuracy = model.evaluate(test_dataset)
        file.write("Test accuracy: {:.2f}%\n".format(round(accuracy * 100, 2)))
        _, accuracy = model.evaluate(val_dataset)
        file.write("Val accuracy: {:.2f}%\n".format(round(accuracy * 100, 2)))

        # Save settings
        file.write("Settings:\n")
        file.write(f"image_size: {image_size}\n")
        file.write(f"filters: {filters}\n")
        file.write(f"depth: {depth}\n")
        file.write(f"kernel_size: {kernel_size}\n")
        file.write(f"patch_size: {patch_size}\n")
        file.write(f"num_classes: {num_classes}\n")

    return file_path



conv_mixer_model = get_conv_mixer_256_8(image_size=image_size, filters=filters, depth=depth, kernel_size=kernel_size,
                                        patch_size=patch_size, num_classes=num_classes)
history, conv_mixer_model = run_experiment(conv_mixer_model)
save_experiment_results(filters=filters, depth=depth, kernel_size=kernel_size,
                                        patch_size=patch_size, num_classes=num_classes, model=conv_mixer_model)

plot_training_history(history, directory)


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


# We first visualize the learned patch embeddings.
patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
visualization_plot(patch_embeddings)

# First, print the indices of the convolution layers that are not
# pointwise convolutions.
for i, layer in enumerate(conv_mixer_model.layers):
    if isinstance(layer, layers.DepthwiseConv2D):
        if layer.get_config()["kernel_size"] == (5, 5):
            print(i, layer)

idx = 26  # Taking a kernel from the middle of the network.

kernel = conv_mixer_model.layers[idx].get_weights()[0]
kernel = np.expand_dims(kernel.squeeze(), axis=2)
visualization_plot(kernel)
