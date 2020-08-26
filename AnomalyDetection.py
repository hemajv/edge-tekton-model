import os, sys
import pickle


class AnomalyDetection(object):
    def __init__(self):
        print("Initializing...")
        self.model_file = os.environ.get("MODEL_FILE", "trained_model.pkl")

        print("Load modelfile: %s" % (self.model_file))
        self.clf = pickle.load(open(self.model_file, "rb"))

    def predict(self, X, feature_names):
        prediction = self.clf.predict(X)

        return prediction[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def forecast_anomalies(self, forecast_df, signal_df):
        detections = []
        for y, yhat_upper, yhat_lower in zip(
            signal_df.y, forecast_df.yhat_upper, forecast_df.yhat_lower
        ):

            if y <= yhat_lower:
                detections.append(1)

            elif y >= yhat_upper:
                detections.append(1)

            else:
                detections.append(0)

        detected_df = forecast_df
        detected_df["predicted_anomaly"] = detections

        return detected_df
