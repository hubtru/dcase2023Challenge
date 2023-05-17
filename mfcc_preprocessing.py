import librosa
import os
import json
import numpy as np


# feature_type: 'mel' or 'mfcc', decides between mfcc or melspectrogram as features
# augment: Boolean, decides whether or not Data Augmentation is applied
# num_augmentation: Number of augmentations per audio file
# augmentation_factor: Magnitude of random noice added
def compute_features(audio_dir, feature_type='mfcc', augment=False, num_augmentations=5, augmentation_factor=0.02):
    # Data structure to store features
    features_data = {}

    # Iterate over audio files
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):  # Adjust the file extension according to your audio files
            filepath = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)  # Load audio file

            if augment:
                # Apply data augmentation
                augmented_features = []
                for _ in range(num_augmentations):
                    augmented_audio = audio + np.random.randn(len(audio)) * augmentation_factor
                    features = extract_features(audio=augmented_audio, sr=sr, feature_type=feature_type)
                    augmented_features.append(features.tolist())

                # Include the original features in the array
                original_features = extract_features(audio=audio, sr=sr, feature_type=feature_type)
                augmented_features.append(original_features.tolist())
                #print(np.shape(augmented_features))
                '''
                augmented_features = np.array(augmented_features)
                augmented_features = augmented_features.reshape(
                    augmented_features.shape[0] * augmented_features.shape[1],
                    augmented_features.shape[2])
                '''
                features_data[filename] = augmented_features
            else:
                # Compute features without augmentation
                features = extract_features(audio=audio, sr=sr, feature_type=feature_type)
                '''
                features = np.array(features)
                features = features.reshape(
                    features.shape[0] * features.shape[1],
                    features.shape[2])
                '''
                #print(np.shape(features))
                features_data[filename] = features

    return features_data


def extract_features(audio, sr, feature_type):
    if feature_type == 'mfcc':
        return librosa.feature.mfcc(y=audio, sr=sr)
    elif feature_type == 'mel':
        return librosa.feature.melspectrogram(y=audio, sr=sr)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}. Supported types are 'mfcc' and 'mel'.")


def save_features(features_data, output_file):
    # Convert NumPy arrays to lists if needed
    features_data_serializable = {}
    for filename, features in features_data.items():
        if isinstance(features, np.ndarray):
            features_data_serializable[filename] = features.tolist()
        else:
            features_data_serializable[filename] = features

    # Save features to a JSON file
    with open(output_file, 'w') as f:
        json.dump(features_data_serializable, f)

    print(f"{output_file} saved")



def load_features_from_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Extract the feature data from the dictionary values
    features_data = list(json_data.values())

    # Convert the feature data to a Numpy array
    features_data_np = np.array(features_data)

    return features_data_np


def save_all_features(audio_dirs, output_files, feature_type='mfcc', augment=False, num_augmentations=5,
                      augmentation_factor=0.02):
    for audio_dir, output_file in zip(audio_dirs, output_files):
        features_data = compute_features(audio_dir, feature_type, augment, num_augmentations, augmentation_factor)
        save_features(features_data, output_file)








