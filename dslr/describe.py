import numpy as np
import matplotlib.pyplot as plot
import warnings
import sys

if len(sys.argv) != 2:
    print("Error: Usage: python describe.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]

try:
    with warnings.catch_wanings():
        warnings.simplefilter("error")
        data = np.loadtxt('')