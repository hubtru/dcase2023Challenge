import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import librosa


def visualize_encoded_data(train_data, test_data):
    # Reshape the data to have two dimensions
    train_data_2d = train_data.reshape(train_data.shape[0], -1)
    test_data_2d = test_data.reshape(test_data.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)
    reduced_train_data = pca.fit_transform(train_data_2d)
    reduced_test_data = pca.transform(test_data_2d)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the reduced encoded data
    axs[0].scatter(reduced_train_data[:, 0], reduced_train_data[:, 1])
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[0].set_title('Train Data Visualization')

    # Plot the reduced new data
    axs[1].scatter(reduced_test_data[:, 0], reduced_test_data[:, 1])
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    axs[1].set_title('Test Data Visualization')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Calculate the cumulative percentage of information explained by the principal components
    explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Create a separate plot for the cumulative information explained
    fig, ax = plt.subplots(figsize=(8, 6))
    num_components = min(10, len(explained_variance_ratio_cumulative))
    ax.bar(range(1, num_components + 1), explained_variance_ratio_cumulative[:num_components], align='center')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title('Cumulative Explained Variance Ratio by Number of Components')

    # Show the plot
    plt.show()


def visualize_melspectrogram(mel_spectrogram):
    # Reshape the mel spectrogram to remove the singleton dimension
    mel_spectrogram = np.squeeze(mel_spectrogram)

    # Plot mel spectrogram without dB conversion
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f')
    plt.title('Mel Spectrogram (dB Conversion)')

    # Show the plot
    plt.show()
