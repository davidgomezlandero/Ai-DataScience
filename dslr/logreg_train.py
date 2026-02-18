import matplotlib.pyplot as pl
import numpy as np
import warnings
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Error: Usage: python logreg_train.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]
if filename != 'datasets/dataset_train.csv':
    print("The dataset for training must be called dataset_train.csv and saved in the directory 'datasets'. Your path must be datasets/dataset_train.csv")
    sys.exit(1)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw_data = pd.read_csv(filename)
except:
    print("Error loading dataset_train.csv")
    sys.exit(1)

titles = raw_data.columns[6:].tolist() #Get column headers from column 6 onwards and convert to list
""" Features used to train the model """
features = ["Herbology", "Ancient Runes","Flying", "Defense Against the Dark Arts",  "Divination", "Charms","History of Magic"]
print(raw_data["Hogwarts House"].value_counts())
raw_data_clean = raw_data.dropna(subset=features + ['Hogwarts House'])#Remove rows with NaN in specific columns

""" Split 80/30 """
np.random.seed(42)#Set random seed for reproducibility
indices = np.random.permutation(len(raw_data_clean))#Create array with indices and shuffle them randomly
split_idx = int(0.8* len(indices))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]



""" Creating and cleaning datasets """
train_data = raw_data_clean.iloc[train_indices].reset_index(drop=True)#Get selected indices and reset them to 0
test_data = raw_data_clean.iloc[test_indices].reset_index(drop=True)

print("Train set:")
print(train_data["Hogwarts House"].value_counts())

# Contar estudiantes por casa en test
print("\nTest set:")
print(test_data["Hogwarts House"].value_counts())

""" Saving test set to use it in accuracy.py """
test_data.to_csv("test.csv", index=False)

""" normalitation functions"""
data_train = train_data.to_numpy()
features_index = [train_data.columns.get_loc(s) for s in features] #Get a list with column positions of the features

def mean_manual(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = []
    for col in range(n_cols):
        col_sum = 0
        for row in range(n_rows):
            col_sum += matrix[row][col]
        means.append(col_sum / n_rows)
    return means

def std_manual(matrix, means):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    stds = []
    for col in range(n_cols):
        squared_diff_sum = 0
        for row in range(n_rows):
            squared_diff_sum += (matrix[row][col] - means[col]) ** 2
        variance = squared_diff_sum / n_rows
        stds.append(variance ** 0.5)
    return stds

""" Normalization """
X = data_train[:, features_index].astype(float)
X_list = X.tolist()
X_mean = mean_manual(X_list)
X_std = std_manual(X_list, X_mean)

for i in range(len(X_list)):
    for j in range(len(X_list[0])):
        X_list[i][j] = (X_list[i][j] - X_mean[j]) / X_std[j]

X = np.array(X_list)

""" Bias """
m = X.shape[0]#Number of data points
X = np.c_[np.ones(m), X]#Create array of 1s with length m and concatenate it with X
""" Sigmoid function """
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
""" Calculate z and call sigmoid function """
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    epsilon = 1e-15
    return (-1 / m) * np.sum(
        y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
    )

""" Gradient descent implementations """
""" Batch GD (Uses all the avalaible data) """
def batch_gradient_descent(X, y, theta, alpha, max_iterations, tolerance=1e-6):
    m = len(y)
    prev_cost = float('inf')
    
    for _ in range(max_iterations):
        h = hypothesis(X, theta)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        
        current_cost = cost_function(X, y, theta)
        cost_diff = abs(prev_cost - current_cost)
        
        if cost_diff < tolerance:
            break
        
        prev_cost = current_cost
    
    return theta

""" Mini-batch GD """
def mini_batch_gradient_descent(X, y, theta, alpha, max_iterations, batch_size=32, tolerance=1e-6):
    m = len(y)
    prev_cost = float('inf')
    
    for _ in range(max_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            end_idx = min(i + batch_size, m)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            batch_m = len(y_batch)
            
            h = hypothesis(X_batch, theta)
            gradient = (1 / batch_m) * np.dot(X_batch.T, (h - y_batch))
            theta -= alpha * gradient
        
        current_cost = cost_function(X, y, theta)
        cost_diff = abs(prev_cost - current_cost)
        
        if cost_diff < tolerance:
            break
        
        prev_cost = current_cost
    
    return theta

""" SGD """
def stochastic_gradient_descent(X, y, theta, alpha, max_iterations, tolerance=1e-6):
    m = len(y)
    prev_cost = float('inf')
    
    for _ in range(max_iterations):
        indices = np.random.permutation(m)
        
        for idx in indices:
            x_i = X[idx:idx+1]
            y_i = y[idx:idx+1]
            
            h = hypothesis(x_i, theta)
            gradient = np.dot(x_i.T, (h - y_i))
            theta -= alpha * gradient
        
        current_cost = cost_function(X, y, theta)
        cost_diff = abs(prev_cost - current_cost)
        
        if cost_diff < tolerance:
            break
        
        prev_cost = current_cost
    
    return theta

def denormalize_theta(theta, X_mean, X_std):
    theta_denorm = np.zeros_like(theta)
    for i in range(1, len(theta)):
        theta_denorm[i] = theta[i] / X_std[i-1]
    theta_denorm[0] = theta[0]
    for i in range(1, len(theta)):
        theta_denorm[0] -= theta[i] * X_mean[i-1] / X_std[i-1]
    return theta_denorm

""" Training """

houses = train_data["Hogwarts House"].unique() #Get house names
alpha = 0.1
max_iterations = 10000
tolerance = 1e-6

"Choose the training method (batch, mini_batch or sgd)"
method = 'batch'

models = {}
models_denorm = {}

for house in houses:
    y = (train_data["Hogwarts House"] == house).astype(int).to_numpy()#Binary vector identifying if the student belongs to the studied house or not
    theta = np.zeros(X.shape[1])#Initialize theta for this house to 0
    
    if method == 'batch':
        theta = batch_gradient_descent(X, y, theta, alpha, max_iterations, tolerance)
    elif method == 'mini_batch':
        theta = mini_batch_gradient_descent(X, y, theta, alpha, max_iterations, batch_size=32, tolerance=tolerance)
    elif method == 'sgd':
        theta = stochastic_gradient_descent(X, y, theta, alpha, max_iterations, tolerance)
    
    models[house] = theta
    theta_denorm = denormalize_theta(theta, X_mean, X_std)
    models_denorm[house] = theta_denorm

""" Save thetas in theta.csv to use it in logreg_predict.py """
with open("theta.csv", "w") as f:
    f.write("house,bias," + ",".join(features) + "\n")
    for house, theta in models_denorm.items():
        f.write(house + "," + ",".join(map(str, theta)) + "\n")

with open("mean.csv", "w") as f:
    f.write(",".join(features) + "\n" + ",".join(map(str,X_mean)) + "\n")
