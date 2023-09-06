import time

from sklearn.ensemble import IsolationForest
import tensorflow as tf

from evaluation import compute_scores
from model_training import normalize_features
from preprocessing import compute_all_features
from settings import start_model
from visualization import visualize_encoded_data


class IsolationForestDetector:
    def __init__(self, contamination=0.05, random_state=None):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def train(self, data):
        self.model.fit(data)

    def detect_anomalies(self, data):
        anomaly_scores = self.model.decision_function(data)
        predictions = self.model.predict(data)
        return anomaly_scores, predictions


def main():
    start_time = time.time()
    model = "IsolationForest"
    gaussion_noise = 0.00
    audio_all, datasets, output_size, feature = start_model()

    # Preprocess data
    data_train, data_test, test_real_classification = compute_all_features(audio_all, datasets=datasets,
                                                                           feature_type=feature, augment=False,
                                                                           num_augmentations=5,
                                                                           augmentation_factor=0.02,
                                                                           output_size=output_size,
                                                                           save=False)
    data_train = normalize_features(data_train, model)
    data_test = normalize_features(data_test, model)

    # Add Noise
    data_train_noise = data_train + gaussion_noise * tf.random.normal(shape=data_train.shape)
    data_test_noise = data_test + gaussion_noise * tf.random.normal(shape=data_test.shape)

    # Train Model
    isolation_forest_detector = IsolationForestDetector(contamination=0.05)
    isolation_forest_detector.train(data_train)

    # Detect anomalies
    anomaly_scores, train_predictions = isolation_forest_detector.detect_anomalies(data_train_noise)
    anomaly_scores, test_predictions = isolation_forest_detector.detect_anomalies(data_test_noise)

    encoded_train_data = data_train
    encoded_test_data = data_test

    encoded_train_data = encoded_train_data.reshape(-1, output_size[0], output_size[1])
    encoded_test_data = encoded_test_data.reshape(-1, output_size[0], output_size[1])
    data_train = data_train.reshape(-1, output_size[0], output_size[1])
    data_test = data_test.reshape(-1, output_size[0], output_size[1])

    # Visualizations after calculations
    visualize_encoded_data(encoded_train_data, encoded_test_data, train_predictions, test_predictions,
                           test_real_classification["anomaly"])
    # visualize_melspectrogram(data_train[0])

    # Evaluation
    compute_scores(test_real_classification["anomaly"], test_predictions)

    # Compilation time
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time} seconds")


if __name__ == '__main__':
    main()
