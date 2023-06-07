import librosa
import os
import json
import numpy as np


# feature_type: 'mel', 'stft' or 'mfcc', decides between mfcc, stft or melspectrogram as features
# augment: Boolean, decides whether or not Data Augmentation is applied
# num_augmentation: Number of augmentations per audio file
# augmentation_factor: Magnitude of random noice added
def compute_features(audio_dir, feature_type, augment=False, num_augmentations=5, augmentation_factor=0.02):
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
                    augmented_features.append(features)

                # Include the original features in the array
                original_features = extract_features(audio=audio, sr=sr, feature_type=feature_type)
                augmented_features.append(original_features)
                features_data[filename] = np.array(augmented_features)
            else:
                # Compute features without augmentation
                features = extract_features(audio=audio, sr=sr, feature_type=feature_type)
                features_data[filename] = features

    return features_data


def compute_all_features(audio_dir, subsets, datasets, feature_type='mfcc', augment=False, num_augmentations=5,
                         augmentation_factor=0.02):
    data_folder = "data_features"
    os.makedirs(data_folder, exist_ok=True)

    for subset in subsets:
        # Get the appropriate subset directories based on the subset parameter
        subset_dirs = [os.path.join(audio_dir, folder, folder2, subset) for folder in os.listdir(audio_dir) for folder2 in os.listdir(os.path.join(audio_dir, folder))]
        print(subset_dirs)
        # Iterate over audio directories in the chosen subset
        for subset_dir, dataset in zip(subset_dirs, datasets):
            features = compute_features(audio_dir=subset_dir, feature_type=feature_type, augment=augment,
                                        num_augmentations=num_augmentations, augmentation_factor=augmentation_factor)
            filename = os.path.join(data_folder, f"{feature_type}_{dataset}_{subset}.json")
            save_features(features, filename)


def extract_features(audio, sr, feature_type):
    if feature_type == 'mfcc':
        return librosa.feature.mfcc(y=audio, sr=sr) # 1000 x 20 x 313
    elif feature_type == 'mel':
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        return librosa.power_to_db(mel_spectrogram, ref=np.max)# 1000 x 128 x 313
    elif feature_type == 'stft':
        return librosa.stft(audio) # 1000 x 1025 x 313
    else:
        raise ValueError(f"Invalid feature type: {feature_type}. Supported types are 'mfcc' and 'mel'.")


def save_features(features_data, output_file):
    # Convert NumPy arrays to lists if needed
    features_data_serializable = {}
    for filename, features in features_data.items():
        if isinstance(features, np.ndarray):
            if np.iscomplexobj(features):
                features = np.abs(features)  # Convert complex values to magnitude
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


def load_all_features(feature_type, subsets, datasets):
    features_train = []
    features_test = []
    data_folder = "data_features"

    for subset in subsets:
        for dataset in datasets:
            json_file = os.path.join(data_folder, f"{feature_type}_{dataset}_{subset}.json")
            features = load_features_from_json(json_file)
            if subset == 'train':
                features_train.append(features)
            elif subset == 'test':
                features_test.append(features)

    # Concatenate all features for each subset
    if features_train:
        features_train = np.concatenate(features_train, axis=0)
    if features_test:
        features_test = np.concatenate(features_test, axis=0)

    np.random.shuffle(features_train)
    np.random.shuffle(features_test)
    return features_train, features_test


