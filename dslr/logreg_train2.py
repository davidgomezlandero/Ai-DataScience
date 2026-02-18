import numpy as np
import warnings
import sys
import pandas as pd

class DataLoader:
    def __init__(self, filename, features):
        self.filename = filename
        self.features = features
    
    def load(self):

def main():
	if len(sys.argv) != 2:
		print("Error: Usage: python logreg_train.py <dataset.csv>")
		sys.exit(1)

	filename = sys.argv[1]
	features = ['Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Charms','Flying']

	if filename != 'datasets/dataset_train.csv':
		print("The dataset for training must be called dataset_train.csv and saved in the directory 'datasets'. Your path must be datasets/dataset_train.csv")
		sys.exit(1)
	try: