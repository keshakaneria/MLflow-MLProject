from sklearn.ensemble import RandomForestClassifier
import mlflow
import pandas as pd
import sys

print("-----------------Entering Train.py!----------------")
data_path = sys.argv[1]
print("Data path = ",data_path)
model_path = sys.argv[2]
print("Model path = ", model_path)
rfc = RandomForestClassifier()
label = 'A'

x = pd.read_csv(data_path)
y = x.pop(label)

mlflow.log_metric("train", 1000.00)
model = RandomForestClassifier()
model = rfc.fit(x,y)
mlflow.sklearn.save_model(model, model_path)
print("-----------------Model Saved!----------------")
