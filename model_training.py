import tensorflow as tf
import numpy as np
from keras.layers import Dense, Dropout, \
    Conv2D, MaxPooling2D, Input, UpSampling2D, Reshape, Flatten
from keras.regularizers import l2
from sklearn.utils import shuffle
from keras.models import Model


class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, l2_reg=0.00, dropout_rate=0.00):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.flatten_layer = Flatten()
        self.encoder = tf.keras.Sequential([
            Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dense(latent_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        ])
        self.dropout = Dropout(dropout_rate)
        self.decoder = tf.keras.Sequential([
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            Dense(input_shape[0], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        ])

    def call(self, x):
        x_flattened = self.flatten_layer(x)
        encoded = self.encoder(x_flattened)
        encoded = self.dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded


def normalize_features(features_data, model):

    if features_data.shape[1] == 1:
        features_data = np.squeeze(features_data, axis=1)

    # Calculate the mean and standard deviation along the axis of each feature
    mean = np.mean(features_data, axis=(0, 2), keepdims=True)
    std = np.std(features_data, axis=(0, 2), keepdims=True)

    # Perform the normalization
    normalized_data = (features_data - mean) / std

    if model == "Autoencoder" or model == "IsolationForest":
        x, height, width = normalized_data.shape
        normalized_data = normalized_data.reshape(x, height * width)

    print(np.shape(normalized_data))
    return normalized_data


def shuffle_data_and_labels(data, labels=None, random_state=None):
    if labels is not None:
        shuffled_data, shuffled_labels = shuffle(data, labels, random_state=random_state)
        return shuffled_data, shuffled_labels
    else:
        shuffled_data = shuffle(data, random_state=random_state)
        return shuffled_data


class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, l2_reg=0.00, dropout_rate=0.00):
        super(ConvolutionalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg), input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            MaxPooling2D(pool_size=(2, 2))
        ])

        self.dropout = Dropout(dropout_rate)

        self.decoder = tf.keras.Sequential([
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded

