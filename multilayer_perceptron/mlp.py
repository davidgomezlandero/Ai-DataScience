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
    pass

if __name__ == '__main__':
    try:
        parse()
        option = sys.argv[1]
        if option == '--dataset':
            split_dataset()
    except Exception as e:
        print(f"Error: {e}")