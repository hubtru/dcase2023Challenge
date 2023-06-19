import librosa
import os
import json
import numpy as np
from scipy.ndimage import zoom
from scipy import interpolate


# feature_type: 'mel', 'stft' or 'mfcc', decides between mfcc, stft or melspectrogram as features
# augment: Boolean, decides whether or not Data Augmentation is applied
# num_augmentation: Number of augmentations per audio file
# augmentation_factor: Magnitude of random noice added
def compute_features(audio_dir, feature_type, augment=False, num_augmentations=5, augmentation_factor=0.02,
                     output_size=(20, 313)):
    # Initialize lists to store features, classifications, and filenames
    features = []
    classifications = []
    filenames = []

    # Iterate over audio files
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_dir, filename)
            audio, sr = librosa.load(filepath, sr=None)

            if augment:
                augmented_features = []
                for _ in range(num_augmentations):
                    augmented_audio = audio + np.random.randn(len(audio)) * augmentation_factor
                    feature = extract_features(audio=augmented_audio, sr=sr, feature_type=feature_type)
                    augmented_features.append(feature)

                original_features = extract_features(audio=audio, sr=sr, feature_type=feature_type,
                                                     output_size=output_size)
                augmented_features.append(original_features)

                features.append(np.array(augmented_features))
            else:
                feature = extract_features(audio=audio, sr=sr, feature_type=feature_type,
                                            output_size=output_size)
                features.append(feature)

            # Extract classification from the filename
            classification = 1 if "anomaly" in filename else 0
            classifications.append(classification)

            filenames.append(filename)

    # Convert features, classifications, and filenames to numpy arrays
    features = np.array(features)
    classifications = np.array(classifications)
    filenames = np.array(filenames)

    return features, classifications, filenames


def compute_all_features(audio_dir, datasets, feature_type='mfcc', augment=False, num_augmentations=5,
                         augmentation_factor=0.02, output_size=(20, 313), save=False):
    data_folder = "data_features"
    os.makedirs(data_folder, exist_ok=True)

    # Initialize arrays for train and test features, classifications, and filenames
    train_features = np.array([])
    test_features = np.array([])
    test_classifications = np.array([])
    test_filenames = np.array([])

    # Iterate over the specified datasets
    for dataset in datasets:
        folder_path = os.path.join(audio_dir, dataset)
        if not os.path.isdir(folder_path):
            continue

        # Get the subdirectories within each folder
        subdirs = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

        # Iterate over the subdirectories
        for subdir in subdirs:
            subset_dir = os.path.join(folder_path, subdir)
            features, classifications, filenames = compute_features(audio_dir=subset_dir, feature_type=feature_type,
                                                                    augment=augment,
                                                                    num_augmentations=num_augmentations,
                                                                    augmentation_factor=augmentation_factor,
                                                                    output_size=output_size)

            if subdir == "train":
                # Concatenate train features
                train_features = np.concatenate((train_features, features), axis=0) if train_features.size else features
            elif subdir == "test":
                # Concatenate test features, classifications, and filenames
                test_features = np.concatenate((test_features, features), axis=0) if test_features.size else features
                test_classifications = np.concatenate((test_classifications, classifications), axis=0) if test_classifications.size else classifications
                test_filenames = np.concatenate((test_filenames, filenames), axis=0) if test_filenames.size else filenames

            if save:
                output_file = os.path.join(data_folder,
                                        f"{feature_type}_{dataset}_{subdir}_{output_size[0]}_{output_size[1]}.json")
                save_features(features, filenames, output_file)

    return train_features, test_features, test_classifications


def extract_features(audio, sr, feature_type, output_size):
    if feature_type == 'mfcc':
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=output_size[0])
        return rescale_features(mfcc, output_size, feature_type)
    elif feature_type == 'mel':
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=output_size[0])
        rescaled_mel = rescale_features(mel_spectrogram, output_size, feature_type)
        return librosa.power_to_db(rescaled_mel, ref=np.max)
    elif feature_type == 'stft':
        stft = librosa.stft(audio)
        return rescale_features(stft, output_size, feature_type)
    elif feature_type == 'fft':
        fft = np.abs(np.fft.fft(audio))
        return rescale_features(fft, output_size, feature_type)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}. Supported types are 'mfcc', 'mel', 'stft', and 'fft'.")


def save_features(features_data, filenames, output_file):
    # Convert NumPy arrays to lists if needed
    features_data_serializable = {filename: features.tolist() if np.iscomplexobj(features) else features
                                  for filename, features in zip(filenames, features_data)}

    # Save features to a JSON file
    with open(output_file, 'w') as f:
        json.dump(features_data_serializable, f, default=complex_to_float)

    print(f"{output_file} saved")


def complex_to_float(obj):
    if isinstance(obj, complex):
        return abs(obj)  # Convert complex values to magnitude
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def load_features_from_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Extract the feature data from the dictionary values
    features_data = list(json_data.values())

    # Convert the feature data to a Numpy array
    features_data_np = np.array(features_data)

    return features_data_np


def load_all_features(feature_type, subsets, datasets, output_size):
    features_train = []
    features_test = []
    data_folder = "data_features"

    for subset in subsets:
        for dataset in datasets:
            json_file = os.path.join(data_folder,
                                     f"{feature_type}_{dataset}_{subset}_{output_size[0]}_{output_size[1]}.json")
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


def rescale_features(features, output_size, feature_type):
    current_size = features.shape
    if feature_type == "mfcc" or feature_type == "mel":

        # Scaling Factor
        scale_factor_time = current_size[1] / output_size[1]

        # Rescaling function for the time axis using cubic spline interpolation
        rescale_time = interpolate.interp1d(
            np.arange(current_size[1]), features, axis=1, kind='cubic'
        )

        # Perform rescaling
        return rescale_time(np.arange(output_size[1]) * scale_factor_time)

    else:
        # Compute the scaling factors for the frequency and time axes
        scale_factor_freq = output_size[0] / current_size[0]
        scale_factor_time = output_size[1] / current_size[1]

        # Perform rescaling with cubic spline interpolation
        rescaled_feature = zoom(features, (scale_factor_freq, scale_factor_time))
        return rescaled_feature

