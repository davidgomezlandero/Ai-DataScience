import matplotlib.pyplot as plt
import warnings
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Error: Usage: python pair_plot.py <dataset.csv>")
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

gryffindor = data[data[:,1] == "Gryffindor"]
ravenclaw = data[data[:,1] == "Ravenclaw"]
slytherin = data[data[:,1] == "Slytherin"]
hufflepuff = data[data[:,1] == "Hufflepuff"]

gryffindor_scores = gryffindor[:,6:].astype(float)
ravenclaw_scores = ravenclaw[:,6:].astype(float)
slytherin_scores = slytherin[:,6:].astype(float)
hufflepuff_scores = hufflepuff[:,6:].astype(float)

scores = data[:, 6:].astype(float)
n_features = len(titles)

fig, axes = plt.subplots(n_features, n_features, figsize=(22, 22))
fig.suptitle('Pair Plot - DSLR Dataset', fontsize=16, y=0.995)

for i in range(n_features):
    for j in range(n_features):
        ax = axes[i, j]
        
        if i == j:
            ax.hist(hufflepuff_scores[:, i], bins=20, alpha=0.4, color='yellow', density=True)
            ax.hist(gryffindor_scores[:, i], bins=20, alpha=0.4, color='red', density=True)
            ax.hist(ravenclaw_scores[:, i], bins=20, alpha=0.4, color='blue', density=True)
            ax.hist(slytherin_scores[:, i], bins=20, alpha=0.4, color='green', density=True)
        else:
            ax.scatter(hufflepuff_scores[:, j], hufflepuff_scores[:, i], 
                      alpha=0.3, color='yellow', s=1)
            ax.scatter(gryffindor_scores[:, j], gryffindor_scores[:, i], 
                      alpha=0.3, color='red', s=1)
            ax.scatter(ravenclaw_scores[:, j], ravenclaw_scores[:, i], 
                      alpha=0.3, color='blue', s=1)
            ax.scatter(slytherin_scores[:, j], slytherin_scores[:, i], 
                      alpha=0.3, color='green', s=1)
        
        if i == n_features - 1:
            ax.set_xlabel(titles[j], fontsize=10, rotation = 45, ha='right')
        if j == 0:
            ax.set_ylabel(titles[i], fontsize=10, rotation = 45, ha='right')
        
        ax.tick_params(labelsize=6)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='yellow', alpha=0.4, label='Hufflepuff'),
    Patch(facecolor='red', alpha=0.4, label='Gryffindor'),
    Patch(facecolor='blue', alpha=0.4, label='Ravenclaw'),
    Patch(facecolor='green', alpha=0.4, label='Slytherin')
]
plt.tight_layout(rect=[0, 0.03, 0.95, 0.99])  # Deja espacio arriba para el título
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.8, 0.99), ncol=4)
plt.savefig("pair_plot.png")
plt.close()