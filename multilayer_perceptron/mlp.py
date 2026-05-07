import matplotlib.pyplot as pl
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import pandas as pd

def parse():
    options = ["--dataset", "--training", "--predict"]
    if len(sys.argv) < 2 or sys.argv[1] not in options:
        raise Exception("wrong usage, this program has three options:\n \
	1. \'--dataset\' -> It will take the second argument (.csv) and split in train.csv and test.csv.\n \
	2. \'--training\' -> It will train the model using trains.csv\n \
	3. \'--predict\' -> It will make predictions using test.csv by default or the third argument(.csv) and put all them in a results.csv file")

def open_dataset(dataset):
	data = pd.read_csv(dataset, header=None)
	return data

def split_dataset():
    if len(sys.argv) != 3:
        raise Exception("--dataset <dataset.csv>")
    try:
        raw_data = open_dataset(sys.argv[2])
    except:
        raise Exception("could not open the .csv file")
    
    #Split 80/20
    np.random.seed(42)
    raw_data_clean = raw_data.dropna()
    train_data, test_data = train_test_split(
		raw_data_clean,
		test_size=0.2,
		random_state=42,
		stratify=raw_data_clean[1]
	)
    train_data.drop(columns=[0]).to_csv("train.csv", index=False, header=False)
    test_data.drop(columns=[0]).to_csv("test.csv", index=False, header=False)


# Helps to check if the dataset is balanced.
def check_balance(dataset):
	class_counts = dataset[0].value_counts() # Especify the diagnosis column
	print("Class distribution:")
	print(class_counts)
	print("\nPercentages:")
	print(class_counts / len(dataset) * 100) 

if __name__ == '__main__':
    try:
        parse()
        option = sys.argv[1]
        if option == '--dataset':
            split_dataset()
        check_balance(open_dataset('test.csv'))
        check_balance(open_dataset('train.csv'))
    except Exception as e:
        print(f"Error: {e}")