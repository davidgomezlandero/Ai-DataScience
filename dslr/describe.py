import numpy
import matplotlib.pyplot as plot
import warnings
import sys
import math
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

titles = raw_data.columns[6:].tolist()
information = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Variance", "Range"]
raw_data_clean = raw_data.iloc[:, 6:].dropna()
data = raw_data_clean.to_numpy()

count = lambda lst: len(lst)
mean = lambda lst: sum(lst)/len(lst)
def std(lst):
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

def min_value(lst):
    min_value = lst[0]
    for i, v in enumerate(lst):
        if v < min_value: min_value = v
    return min_value

def perc25(lst):
    sorted_list = sorted(lst)
    return sorted_list[int(0.25 * (len(lst) - 1))]

def perc50(lst):
    sorted_list = sorted(lst)
    n = len(lst)
    if n % 2 == 0: return ((sorted_list[n//2 - 1] + sorted_list[n//2]) / 2)
    else: return (sorted_list[n//2])

def perc75(lst):
    sorted_list = sorted(lst)
    return sorted_list[int(0.75 * (len(lst) - 1))]

def max_value(lst):
    sorted_list = sorted(lst)
    return (sorted_list[-1])

def variance(lst):
	m = mean(lst)
	return sum((x - m) ** 2 for x in lst) / len(lst)

def range_value(lst):
	sorted_list = sorted(lst)
	return sorted_list[-1] - sorted_list[0]

funciones = [count, mean, std, min_value, perc25, perc50, perc75, max_value, variance, range_value]

col_widths = []
for title in titles:
    col_widths.append(max(len(title), 15))

print(f'{"":8}', end='')
for i, title in enumerate(titles):
    print(f'{title:>{col_widths[i]}}  ', end='')
print()

for i, v in enumerate(information):
    print(f'{v:8}', end='')
    for j in range(data.shape[1]):
        result = funciones[i](data[:,j])
        print(f'{result:>{col_widths[j]}.6f}  ', end='')
    print()