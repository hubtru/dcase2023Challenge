from visualization import visualize_audio_file

import subprocess
import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sounddevice as sd


def play_audio_file(file_path):
    try:
        # Use the 'start' command in Windows to open the default media player
        subprocess.Popen(['start', file_path], shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


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

    # Randomly select 4 more files without repetition
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
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(15, 5 * num_rows),
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

    plt.subplots_adjust(hspace=0.6)  # Adjust the vertical spacing between subplots
    plt.show()


path2 = r'C:\Users\HABELS.COMPUTACENTER\Downloads\dcase_training_data'
# play_audio_file(path)
# visualize_audio_file(path)


def visualize_datasets(path_all):
    folder_path = get_train_paths(path_all)
    for path in folder_path:
        file_name = select_audio_files(path)
        visualize_audio_files(file_name)


visualize_datasets(path2)
