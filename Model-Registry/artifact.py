from sklearn.ensemble import RandomForestClassifier
import mlflow
import pandas as pd

print("-----------------Entering Artifact.py!----------------")

# Storing artifact locally
# mlflow.set_tracking_uri("./mlruns")

#Storing artifact remotely
mlflow.set_tracking_uri("mysql://sql5517356:E3hQNUZC54@sql5.freesqldatabase.com:3306/sql5517356")
print("Connected to artifact database")

data_path = "..//train.csv"
model_path = "artifact-result"

with mlflow.start_run():
    rfc = RandomForestClassifier()

    print("Reading CSV")
    label = 'A'
    x = pd.read_csv(data_path)
    y = x.pop(label)
    model = RandomForestClassifier()
    model = rfc.fit(x,y)

    print("Logging Model via log_model")
    # mlflow.sklearn.log_model(sk_model=model, artifact_path="train-model1", registered_model_name="reg-train-model1")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="train-model2")
    mlflow.log_metric("register", 1000.00)
    print("Model Registered")

    mlflow.log_metric("artifact", 2000.00)
    mlflow.log_artifact("artifact.py")
    print("Metrics & Artifacts Added")

    # mlflow.sklearn.save_model(model, model_path)
    print("-----------------Model Saved!----------------")

### Adding an MLflow Model to the Model Registry ###

# from sklearn.ensemble import RandomForestClassifier
# import mlflow
# import pandas as pd
# from mlflow import MlflowClient
# from pprint import pprint

# print("-----------------Registering Model----------------")
# mlflow.set_tracking_uri("mysql://sql5517356:E3hQNUZC54@sql5.freesqldatabase.com:3306/sql5517356")
# print("Connected to backend database")

# with mlflow.start_run() as run:
#     rfc = RandomForestClassifier()

#     label = 'A'
#     x = pd.read_csv("../train.csv")
#     y = x.pop(label)
#     model = rfc.fit(x,y)

#     print("Logging Model via log_model")
#     mlflow.sklearn.log_model(sk_model=model, artifact_path="train-model1", registered_model_name="reg-train-model1")
#     mlflow.log_metric("register", 1000.00)
#     print("Model Registered")

# print("-----------------List of all Models----------------")
# client = MlflowClient()
# for rm in client.list_registered_models():
#     pprint(dict(rm), indent=4)

# print("-----------------Train Model registered!----------------")
