import librosa
import os
import json
import numpy as np


# augment: Boolean, decides whether or not Data Augmentation is applied
# num_augmentation: Number of augmentations per audio file
# augmentation_factor: Magnitude of random noice added
def compute_mfccs(audio_dir, augment=False, num_augmentations=5, augmentation_factor=0.02):
    # Data structure to store MFCCs
    mfccs_data = {}

    # Iterate over audio files
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):  # Adjust the file extension according to your audio files
            filepath = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)  # Load audio file

            if augment:
                # Apply data augmentation
                augmented_mfccs = []
                for _ in range(num_augmentations):
                    augmented_audio = audio + np.random.randn(len(audio)) * augmentation_factor
                    mfccs = librosa.feature.mfcc(y=augmented_audio, sr=sr)  # Compute MFCCs
                    augmented_mfccs.append(mfccs.tolist())
                mfccs_data[filename] = augmented_mfccs
            else:
                # Compute MFCCs without augmentation
                mfccs = librosa.feature.mfcc(y=audio, sr=sr)  # Compute MFCCs
                mfccs_data[filename] = [mfccs.tolist()]

    return mfccs_data


def save_mfccs(mfccs_data, output_file):
    # Save MFCCs to a JSON file
    with open(output_file, 'w') as f:
        json.dump(mfccs_data, f)

        def load_mfccs_from_json(json_file):
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            # Extract the MFCC data from the dictionary values
            mfccs_data = list(json_data.values())

            # Convert the MFCC data to a Numpy array
            mfccs_data_np = np.array(mfccs_data)

            return mfccs_data_np


def save_all_mfccs(audio_dir, output_file):
    for i in range(len(audio_dir)):
        mfccs_data = compute_mfccs(audio_dir[i])  # Compute MFCCs
        save_mfccs(mfccs_data, output_file[i])  # Save MFCCs


def load_mfccs_from_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Extract the MFCC data from the dictionary values
    mfccs_data = list(json_data.values())

    # Convert the MFCC data to a Numpy array
    mfccs_data_np = np.array(mfccs_data)

    return mfccs_data_np
