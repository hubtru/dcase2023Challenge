from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, contamination=0.05, random_state=None):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def train(self, data):
        self.model.fit(data)

    def detect_anomalies(self, data):
        anomaly_scores = self.model.decision_function(data)
        predictions = self.model.predict(data)
        return anomaly_scores, predictions
