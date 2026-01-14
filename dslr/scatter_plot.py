import numpy
import matplotlib.pyplot as plt
import warnings
import sys
import math
import pandas as pd

if len(sys.argv) != 2:
    print("Error: Usage: python scatter_plot.py <dataset.csv>")
    exit(1)

filename = sys.argv[1]
if filename != 'datasets/dataset_train.csv':
    print("The dataset for training must be called dataset_train.csv and saved in the directory 'datasets'. Your path must be datasets/dataset_train.csv")
    exit(1)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw_data = pd.read_csv(filename)
except:
    print("Error loading dataset_train.csv")
    exit(1)

titles = raw_data.columns[6:].tolist()
raw_data_clean = raw_data.dropna()
data = raw_data_clean.to_numpy()

scores = data[:, 6:].astype(float)

def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    denominator_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0
    
    return numerator / (denominator_x * denominator_y)

max_corr = -1
feature1_idx = 0
feature2_idx = 0

for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        corr_value = abs(correlation(scores[:, i], scores[:, j]))
        if corr_value > max_corr:
            max_corr = corr_value
            feature1_idx = i
            feature2_idx = j

feature1 = titles[feature1_idx]
feature2 = titles[feature2_idx]
corr_final = correlation(scores[:, feature1_idx], scores[:, feature2_idx])

plt.figure(figsize=(10, 6))
plt.scatter(scores[:, feature1_idx], scores[:, feature2_idx], alpha=0.5)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Scatter Plot: {feature1} vs {feature2}\nCorrelation: {corr_final:.4f}')
plt.grid(True, alpha=0.3)
plt.show()