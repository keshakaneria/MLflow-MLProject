### Adding an MLflow Model to the Model Registry ###

from sklearn.ensemble import RandomForestClassifier
import mlflow
import pandas as pd
from mlflow import MlflowClient
from pprint import pprint

print("-----------------Registering Model----------------")
mlflow.set_tracking_uri("mysql://sql10516588:fG9iHVJpbU@sql10.freesqldatabase.com:3306/sql10516588")
print("Connected to database")

with mlflow.start_run() as run:
    rfc = RandomForestClassifier()

    label = 'A'
    x = pd.read_csv("../train.csv")
    y = x.pop(label)
    model = rfc.fit(x,y)

    print("Logging Model via log_model")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="train-model1", registered_model_name="reg-train-model1")
    print("Model Registered")
    print("Logging Model via register_model")
    # Change the run_id according to your model in Projects/mlruns/0/<main entry point's id>
    result = mlflow.register_model("../Projects/runs:/67d5036cbbfc4e17ae283014276fdc4d/train-model2", "reg-train-model2")
    print("Done")

print("-----------------Fetching one Model----------------")
print("--")
model_uri = f"models:/reg-train-model1/3"
loaded_model = mlflow.sklearn.load_model(model_uri)

print("-----------------List of all Models----------------")
client = MlflowClient()
for rm in client.list_registered_models():
    pprint(dict(rm), indent=4)

print("-----------------Train Model registered!----------------")
