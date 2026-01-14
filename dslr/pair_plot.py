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