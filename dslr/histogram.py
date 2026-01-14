import numpy
import matplotlib.pyplot as plt
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

mean = lambda lst: sum(lst)/len(lst)
def std(lst):
    m = mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

course_pos = 0
tmp_std = float("inf")

for i in range(len(titles)):
    tmp_gryffindor_mean = mean(gryffindor_scores[:,i])
    tmp_ravenclaw_mean = mean(ravenclaw_scores[:,i])
    tmp_slytherin_mean = mean(slytherin_scores[:,i])
    tmp_hufflepuff_mean = mean(hufflepuff_scores[:,i])
    
    if tmp_std > std([tmp_slytherin_mean,tmp_gryffindor_mean,tmp_hufflepuff_mean,tmp_ravenclaw_mean]):
        tmp_std = std([tmp_slytherin_mean,tmp_gryffindor_mean,tmp_hufflepuff_mean,tmp_ravenclaw_mean])
        course_pos = i
        
plt.figure(figsize=(10, 6))
plt.hist(hufflepuff_scores[:, course_pos], bins=20, alpha=0.4, label='Hufflepuff', color='yellow')
plt.hist(gryffindor_scores[:, course_pos], bins=20, alpha=0.4, label='Gryffindor', color='red')
plt.hist(ravenclaw_scores[:, course_pos], bins=20, alpha=0.4, label='Ravenclaw', color='blue')
plt.hist(slytherin_scores[:, course_pos], bins=20, alpha=0.4, label='Slytherin', color='green')

plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Distribution of {titles[course_pos]} scores by House')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()