import matplotlib.pyplot as pl
import numpy as np
import warnings
import sys
import pandas as pd

def parse():
    options = ["--dataset", "--training", "--predict"]
    if len(sys.argv) < 2 or sys.argv[1] not in options:
        raise Exception("wrong usage, this program has three options:\n \
	1. \'--dataset\' -> It will take the second argument (.csv) and split in train.csv and test.csv.\n \
	2. \'--training\' -> It will train the model using trains.csv\n \
	3. \'--predict\' -> It will make predictions using test.csv by default or the third argument(.csv) and put all them in a results.csv file")

def split_dataset():
    if len(sys.argv) != 3:
        raise Exception("--dataset <datset.csv>")
    try:
        raw_data = pd.read_csv(sys.argv[2], header=None)
    except:
        raise Exception("could not open the .csv file")
    
    #Split 80/20
    np.random.seed(42)
    raw_data_clean = raw_data.dropna()
    indexes = np.random.permutation(len(raw_data_clean))
    split_idx = int(0.8 * len(indexes))
    train_data = raw_data_clean.iloc[indexes[:split_idx]].reset_index(drop=True)
    test_data = raw_data_clean.iloc[indexes[split_idx:]].reset_index(drop=True)
    train_data.drop(columns=[0]).to_csv("train.csv", index=False, header=False)
    test_data.drop(columns=[0]).to_csv("test.csv", index=False, header=False)
    

if __name__ == '__main__':
    try:
        parse()
        option = sys.argv[1]
        if option == '--dataset':
            split_dataset()
    except Exception as e:
        print(f"Error: {e}")