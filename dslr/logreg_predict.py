import numpy as np
import warnings
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Error: Usage: python logreg_predict.py <dataset.csv>")
    exit(1)

filename = sys.argv[1]

""" Load dataset """
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw_data = pd.read_csv(filename)
except:
    print(f"Error loading {filename}")
    exit(1)

""" Load theta.csv """
try:
    theta_data = pd.read_csv("theta.csv")
except:
    print("Error: theta.csv not found. Please run logreg_train.py first.")
    exit(1)

features = theta_data.columns[2:].tolist()
houses = theta_data['house'].tolist()



for feature in features:
    if feature not in raw_data.columns:
        print(f"Error: Feature '{feature}' not found in {filename}")
        exit(1)

indices = raw_data['Index'].tolist()

# Extraer características
features_index = [raw_data.columns.get_loc(s) for s in features]

X = raw_data[features].copy()

for i,feature in enumerate(features):
	X[feature] = X[feature].fillna(theta_data.iloc[4,i+2])
X = X.to_numpy().astype(float)
m = X.shape[0]
X = np.c_[np.ones(m), X]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_probability(X, theta):
    return sigmoid(np.dot(X, theta))

predictions = []

for i in range(m):
    x_sample = X[i]
    probabilities = {}

    for idx, house in enumerate(houses):
        theta = theta_data.iloc[idx, 1:].values.astype(float)
        prob = predict_probability(x_sample, theta)
        probabilities[house] = prob
    predicted_house = max(probabilities, key=probabilities.get)
    predictions.append(predicted_house)

with open("houses.csv", "w") as f:
    f.write("Index,Hogwarts House\n")
    for idx, house in zip(indices, predictions):
        f.write(f"{int(idx)},{house}\n")
print("The result is saved in houses.csv")