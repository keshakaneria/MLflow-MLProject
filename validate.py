from sklearn.ensemble import RandomForestClassifier
import mlflow
import pandas as pd
import sys

print("-----------------Entering Validate.py!----------------")
data_path = sys.argv[1]
print("Data path = ", data_path)
model_path = sys.argv[2]
print("Model path = ", model_path)

label = 'A'

x = pd.read_csv(data_path)
y = x.pop(label)

model = mlflow.sklearn.load_model(model_path)
accuracy = model.score(x,y)
print("Accuracy = ",accuracy)
