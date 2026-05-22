import pandas as pd
import numpy as np
import sys

def open_dataset(dataset):
    data = pd.read_csv(dataset, header=None)
    return data

def networks_conf():
    args = sys.argv[2:]
    networks = []
    current_network = []
    
    for arg in args:
        if arg == '--layer' and current_network:
            networks.append(current_network)
            current_network = [arg]
        else:
            current_network.append(arg)
    if current_network:
        networks.append(current_network)
    
    return networks

def standardize(X, mean=None, std=None):
    """
    Applies Z-score normalization. 
    If mean and std are not provided, it calculates them from the dataset X.
    """
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Prevent division by zero for columns that have zero variance
        std = np.where(std == 0, 1e-15, std)
        
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def check_balance(dataset):
    class_counts = dataset[1].value_counts()
    print("Class distribution:")
    print(class_counts)
    print("\nPercentages:")
    print(class_counts / len(dataset) * 100)
    print()
    
def calculate_metrics(y_true, y_pred):
    y_t = np.argmax(y_true, axis=1)
    y_p = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate((y_t, y_p)))
    precisions = []
    recalls = []
    f1s = []
    
    for c in classes:
        tp = np.sum((y_p == c) & (y_t == c))
        fp = np.sum((y_p == c) & (y_t != c))
        fn = np.sum((y_p != c) & (y_t == c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    acc = np.mean(y_t == y_p)
    return acc, np.mean(precisions), np.mean(recalls), np.mean(f1s)