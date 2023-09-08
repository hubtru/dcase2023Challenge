import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import librosa
import librosa.display
import os
from matplotlib.widgets import Button
import sounddevice as sd
import random
import pickle
import seaborn as sns
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import pandas as pd


def visualize_encoded_data(train_data, test_data, train_classes, test_classes, test_true_classes):
    # Reshape the data to have two dimensions
    train_data_2d = train_data.reshape(train_data.shape[0], -1)
    test_data_2d = test_data.reshape(test_data.shape[0], -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)
    reduced_train_data = pca.fit_transform(train_data_2d)
    reduced_test_data = pca.transform(test_data_2d)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the reduced train data
    axs[0].scatter(reduced_train_data[:, 0], reduced_train_data[:, 1], c=train_classes)
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[0].set_title('Train Data Visualization')

    # Plot the reduced test data
    axs[1].scatter(reduced_test_data[:, 0], reduced_test_data[:, 1], c=test_classes)
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    axs[1].set_title('Test Data Visualization')

    # Calculate true positive, false positive, true negative, and false negative
    true_positive = np.logical_and(test_classes == 1, test_true_classes == 1)
    false_positive = np.logical_and(test_classes == 1, test_true_classes == 0)
    true_negative = np.logical_and(test_classes == 0, test_true_classes == 0)
    false_negative = np.logical_and(test_classes == 0, test_true_classes == 1)

    # Mark true positive, false positive, true negative, and false negative in the test data plot
    axs[1].scatter(reduced_test_data[true_positive, 0], reduced_test_data[true_positive, 1], c='green', label='True Positive')
    axs[1].scatter(reduced_test_data[false_positive, 0], reduced_test_data[false_positive, 1], c='red', label='False Positive')
    axs[1].scatter(reduced_test_data[true_negative, 0], reduced_test_data[true_negative, 1], c='blue', label='True Negative')
    axs[1].scatter(reduced_test_data[false_negative, 0], reduced_test_data[false_negative, 1], c='orange', label='False Negative')

    # Add legend
    axs[1].legend()

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


def visualize_audio_length(audio_dir):
    audio_lengths = np.array([])

    for folder in os.listdir(audio_dir):
        folder_path = os.path.join(audio_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Get the subdirectories within each folder
        subdirs = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

        # Iterate over the subdirectories
        for subdir in subdirs:
            subset_dir = os.path.join(folder_path, subdir)
            for filename in os.listdir(subset_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(subset_dir, filename)
                    audio, sr = librosa.load(file_path)
                    duration = librosa.get_duration(y=audio, sr=sr)
                    audio_lengths = np.append(audio_lengths, duration)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(audio_lengths)), audio_lengths)
    plt.xlabel('Audio Index')
    plt.ylabel('Length (seconds)')
    plt.title('Audio Length Comparison')
    plt.tight_layout()
    plt.show()


def visualize_datasets(path_all):
    folder_path = get_train_paths(path_all)
    for path in folder_path:
        file_name = select_audio_files(path)
        visualize_audio_files(file_name)


def select_audio_files(folder_path):
    files = os.listdir(folder_path)
    num_files = len(files)

    # Ensure that at least 10 files exist in the folder
    if num_files < 10:
        raise ValueError("There are not enough files in the folder.")

    # Select the first and last file
    selected_files = [os.path.join(folder_path, files[0]), os.path.join(folder_path, files[-1])]

    # Remove the first and last file from the list
    remaining_files = files[1:-1]

    # Randomly select 3 more files without repetition
    random_files = random.sample(remaining_files, min(3, len(remaining_files)))
    for file in random_files:
        selected_files.append(os.path.join(folder_path, file))
        remaining_files.remove(file)

    return selected_files


def get_train_paths(root_path):
    train_paths = []
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        train_folder_path = os.path.join(folder_path, 'train')
        if os.path.isdir(train_folder_path):
            train_paths.append(train_folder_path)
    return train_paths


def play_audio(event):
    audio, sr = event.inaxes.audio_data
    sd.play(audio, sr)


def visualize_audio_files(file_paths):
    num_files = len(file_paths)

    # Calculate the number of rows needed for the visuals
    num_rows = num_files

    # Adjust subplot parameters to increase height ratio between rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(18, 5 * num_rows),
                             gridspec_kw={'height_ratios': [3] * num_rows})

    play_buttons = []  # List to store the play buttons
    substring = file_paths[0].split('dcase_training_data\\')[1]
    end_index = substring.index('\\train')
    dataset_name = substring[:end_index]

    fig.suptitle(f'Audio Visualization of {dataset_name} Dataset', fontsize=16, fontweight='bold')

    for i, file_path in enumerate(file_paths):
        # Load the audio file
        audio, sr = librosa.load(file_path)

        # Extract the desired part of the filename as the title
        file_name = os.path.basename(file_path)
        title = file_name.split("normal_")[-1]  # Extract the part after "normal_"

        # Calculate the row index for the current file
        row_idx = i

        # Plot the waveform
        axes[row_idx, 0].set_title('Waveform - {}'.format(title), pad=5)  # Add padding to the title
        axes[row_idx, 0].set_xlabel('Time (s)', labelpad=5)  # Add padding to the x-axis label
        axes[row_idx, 0].set_ylabel('Amplitude')
        axes[row_idx, 0].audio_data = audio, sr  # Store audio data in axes for playback
        duration = len(audio) / sr  # Calculate the duration of the audio in seconds
        time = np.linspace(0, duration, len(audio))  # Create the time array
        axes[row_idx, 0].plot(time, audio)
        play_button = Button(axes[row_idx, 0], '▶️')  # Use a play symbol
        play_button.on_clicked(play_audio)
        play_buttons.append(play_button)  # Add the button to the list

        # Compute the mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot the MFCC with normalization
        axes[row_idx, 1].set_title('MFCC', pad=5)  # Add padding to the title
        axes[row_idx, 1].set_xlabel('Time', labelpad=0)  # Add padding to the x-axis label
        axes[row_idx, 1].set_ylabel('MFCC Coefficients')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[row_idx, 1])
        mfcc_norm = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize the MFCC coefficients
        librosa.display.specshow(mfcc_norm, sr=sr, x_axis='time', ax=axes[row_idx, 1])

        # Plot the Mel Spectrogram
        axes[row_idx, 2].set_title('Mel Spectrogram', pad=5)  # Add padding to the title
        axes[row_idx, 2].set_xlabel('Time', labelpad=0)  # Add padding to the x-axis label
        axes[row_idx, 2].set_ylabel('Mel Frequency')
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[row_idx, 2])

        # Plot the STFT as Mel Spectrogram
        axes[row_idx, 3].set_title('STFT as Mel Spectrogram', pad=5)  # Add padding to the title
        axes[row_idx, 3].set_xlabel('Time', labelpad=0)  # Add padding to the x-axis label
        axes[row_idx, 3].set_ylabel('Frequency')
        stft = librosa.stft(audio)
        stft_mel_spec = librosa.amplitude_to_db(np.abs(stft))
        librosa.display.specshow(stft_mel_spec, sr=sr, x_axis='time', y_axis='log', ax=axes[row_idx, 3])
        '''
        # Plot the FFT Spectrogram
        axes[row_idx, 4].set_title('FFT Spectrogram', pad=5)  # Add padding to the title
        axes[row_idx, 4].set_xlabel('Time', labelpad=0)  # Add padding to the x-axis label
        axes[row_idx, 4].set_ylabel('Frequency')
        fft_spec = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio), d=1 / sr)
        axes[row_idx, 4].plot(freqs, fft_spec)
        axes[row_idx, 4].set_xlim(0, sr / 2)  # Show only positive frequencies
        '''
    plt.subplots_adjust(hspace=0.6)  # Adjust the vertical spacing between subplots
    plt.show()


def plot_training_history(history, save_path=None, file_name=None):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    save_file_path = os.path.join(save_path, file_name)
    plt.savefig(save_file_path)

    # Show the plots
    # plt.show()
