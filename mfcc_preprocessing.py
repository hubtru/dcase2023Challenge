import librosa
import os
import json


def compute_mfccs(audio_dir):
    # Data structure to store MFCCs
    mfccs_data = {}

    # Iterate over audio files
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):  # Adjust the file extension according to your audio files
            filepath = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)  # Load audio file
            mfccs = librosa.feature.mfcc(y=audio, sr=sr)  # Compute MFCCs
            mfccs_data[filename] = mfccs.tolist()  # Store MFCCs in data structure

    return mfccs_data


def save_mfccs(mfccs_data, output_file):
    # Save MFCCs to a JSON file
    with open(output_file, 'w') as f:
        json.dump(mfccs_data, f)
