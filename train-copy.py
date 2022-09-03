from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow import MlflowClient
import pandas as pd
import sys

print("-----------------Entering Train.py!----------------")
mlflow.set_tracking_uri("mysql://sql10516588:fG9iHVJpbU@sql10.freesqldatabase.com:3306/sql10516588")

with mlflow.start_run() as run:
    rfc = RandomForestClassifier()

    label = 'A'
    x = pd.read_csv("../train.csv")
    y = x.pop(label)
    model = RandomForestClassifier()
    model = rfc.fit(x,y)

    mlflow.log_metric("train", 1000.00)
    print("Metrics Added")

    mlflow.sklearn.log_model(sk_model=model, artifact_path="train-model")
    client = MlflowClient()
    client.create_registered_model("reg-train-model")
    print("Model named")

print("-----------------Train Model registered!----------------")
