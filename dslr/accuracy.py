import sys
import pandas as pd
from sklearn.metrics import accuracy_score

if len(sys.argv) < 2:
    print("Error: Usage: python accuracy.py <predicted_labels.csv> [true_labels.csv]")
    print("If true_labels.csv is not provided, test.csv will be used by default")
    exit(1)

predicted_file = sys.argv[1]
true_file = sys.argv[2] if len(sys.argv) == 3 else "test.csv"

try:
    true_data = pd.read_csv(true_file)
except:
    print(f"Error loading {true_file}")
    if true_file == "test.csv":
        print("Make sure you have run logreg_train.py to generate test.csv")
    exit(1)

try:
    predicted_data = pd.read_csv(predicted_file)
except:
    print(f"Error loading {predicted_file}")
    exit(1)

if 'Hogwarts House' not in true_data.columns:
    print(f"Error: 'Hogwarts House' column not found in {true_file}")
    exit(1)

if 'Hogwarts House' not in predicted_data.columns:
    print(f"Error: 'Hogwarts House' column not found in {predicted_file}")
    exit(1)

if 'Index' in true_data.columns and 'Index' in predicted_data.columns:
    merged = pd.merge(true_data[['Index', 'Hogwarts House']], 
                      predicted_data[['Index', 'Hogwarts House']], 
                      on='Index', 
                      suffixes=('_true', '_pred'))
    
    if len(merged) == 0:
        print("Error: No matching indices found between files")
        exit(1)
    
    y_true = merged['Hogwarts House_true'].values
    y_pred = merged['Hogwarts House_pred'].values
else:
    if len(true_data) != len(predicted_data):
        print(f"Error: Different number of samples. True: {len(true_data)}, Predicted: {len(predicted_data)}")
        exit(1)
    
    y_true = true_data['Hogwarts House'].values
    y_pred = predicted_data['Hogwarts House'].values

accuracy = accuracy_score(y_true, y_pred)

print(f"ACCURACY:           {accuracy * 100:.2f}%")
