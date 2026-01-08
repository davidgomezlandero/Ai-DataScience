import numpy
import matplotlib.pyplot as plot
import warnings
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Error: Usage: python describe.py <dataset.csv>")
    exit(1)

filename = sys.argv[1]
if filename != 'datasets/dataset_train.csv':
    print("The dataset for training must be called dataset_train.csv and saved in the directory \'datasets\'. Your path must be datasets/dataset_train.csv")
    exit(1)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw_data = pd.read_csv(filename)
except:
    print("Error loading dataset_train.csv")
    exit(1)

data = raw_data.iloc[:, 6:].to_numpy()

for i,l in enumerate(data):
    print(l)
    if i > 4: break