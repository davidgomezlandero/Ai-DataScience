# DSLR - Data Science Logistic Regression

## 📋 Table of Contents
- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Understanding the Problem](#understanding-the-problem)
- [Statistical Analysis - describe.py](#statistical-analysis---describepy)
- [Data Visualization](#data-visualization)
- [Feature Analysis & Selection](#feature-analysis--selection)
- [The Algorithm - Logistic Regression](#the-algorithm---logistic-regression)
- [My Implementation Details](#my-implementation-details)
- [Training Process](#training-process)
- [Prediction System](#prediction-system)
- [Visualization & Analysis](#visualization--analysis)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Key Learnings](#key-learnings)
- [Files Structure](#files-structure)
- [Future Improvements](#future-improvements)
- [References](#references)

## 🎯 Overview

**DSLR** is my second machine learning project at École 42. The goal is to implement a logistic regression classifier from scratch to predict which Hogwarts house a student belongs to based on their academic scores, using gradient descent optimization.

**The Challenge**:
- Build multi-class classification from scratch using only Python, Pandas, NumPy, and Matplotlib
- No scikit-learn or high-level ML libraries
- Understand every line of math and code
- Implement statistical analysis tools manually (recreate pandas `describe()`)

**What I Built**:
1. A statistical analysis tool that mimics pandas `describe()`
2. Data visualization tools (histogram, scatter plot, pair plot)
3. A training system that learns optimal weights saved to `theta.csv` and `mean.csv`
4. A prediction system that classifies students into Hogwarts houses

## 📐 Mathematical Foundation

### The Logistic Model

Unlike linear regression, logistic regression models probabilities using the sigmoid function:

```
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))

where: z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Where:
- **σ(z)** = sigmoid function (outputs probability between 0 and 1)
- **θ** = parameters to learn
- **x** = input features (student scores)
- **z** = linear combination of features

**In my project**:
```
P(Gryffindor|scores) = σ(θ₀ + θ₁×Herbology + θ₂×DADA + ...)
```

### The Sigmoid Function

The sigmoid function converts any real number to a probability:

```
         1
σ(z) = ─────────
       1 + e^(-z)

Properties:
- σ(0) = 0.5
- σ(+∞) → 1
- σ(-∞) → 0
- Always between 0 and 1
```

**Why sigmoid?**
- Maps predictions to probabilities
- Smooth and differentiable (needed for gradient descent)
- Natural interpretation as likelihood

### The Cost Function - Log Loss (Cross-Entropy)

For binary classification:

```
J(θ) = -1/m × Σᵢ₌₁ᵐ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Where:
- m = number of training examples
- yᵢ = actual label (0 or 1)
- ŷᵢ = predicted probability = σ(z)
```

**Why cross-entropy and not MSE?**
- Convex function → guaranteed global minimum
- Penalizes confident wrong predictions very heavily
- Mathematically compatible with sigmoid (clean gradient)

### Gradient Descent Update Rules

```
θⱼ := θⱼ - α × (1/m) × Σᵢ₌₁ᵐ (σ(z^(i)) - y^(i)) × x^(i)_j

Where:
- α = learning rate
- m = number of examples
- σ(z^(i)) = prediction for example i
- y^(i) = actual label for example i
- x^(i)_j = feature j of example i
```

**Interesting**: This formula looks identical to linear regression's update rule, but `σ(z)` replaces `z` as the prediction.

### Multi-Class Classification - One-vs-All

For 4 houses, I train 4 independent binary classifiers:

```
1. Is this student Gryffindor?   (1 = yes, 0 = no)
2. Is this student Slytherin?    (1 = yes, 0 = no)
3. Is this student Ravenclaw?    (1 = yes, 0 = no)
4. Is this student Hufflepuff?   (1 = yes, 0 = no)

Final prediction: argmax(P₁, P₂, P₃, P₄)
→ The house with the highest probability wins
```

## 🔍 Understanding the Problem

### The Dataset

**Input**: Student academic performance across 13 magical subjects  
**Output**: Hogwarts House (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)  
**Training samples**: 1600 students  
**Test samples**: 400 students  

### Sample Data

```csv
Index,Hogwarts House,First Name,Last Name,Birthday,Best Hand,Arithmancy,Astronomy,Herbology,...
218,Gryffindor,Jung,Layman,2001-04-20,Right,71739.0,559.13,-5.13,-5.59,3.42,...
591,Slytherin,Rufus,Sizemore,2000-03-25,Right,45136.0,-511.05,-3.79,5.11,-7.55,...
1327,Ravenclaw,Augusta,Graham,1998-01-01,Right,32682.0,-507.13,4.82,5.07,2.92,...
```

**Features (13 courses)**:
- Arithmancy
- Astronomy
- Herbology
- Defense Against the Dark Arts
- Divination
- Muggle Studies
- Ancient Runes
- History of Magic
- Transfiguration
- Potions
- Care of Magical Creatures
- Charms
- Flying

**Challenges**:
- Missing values (NaN) scattered across features
- Very different scales (Arithmancy: ~50000, Herbology: ~-5)
- 13 dimensions with possible redundant features
- 4 classes to classify simultaneously

## 📊 Statistical Analysis - describe.py

### My Implementation

I recreated pandas' `describe()` function entirely from scratch using only Python's `math` module:

```python
import math
import pandas as pd

def count(series):
    return sum(1 for x in series if not pd.isna(x))

def mean(series):
    values = [x for x in series if not pd.isna(x)]
    return sum(values) / len(values) if values else float('nan')

def std(series):
    values = [x for x in series if not pd.isna(x)]
    m = mean(series)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def min_val(series):
    values = [x for x in series if not pd.isna(x)]
    return min(values) if values else float('nan')

def max_val(series):
    values = [x for x in series if not pd.isna(x)]
    return max(values) if values else float('nan')

def percentile(series, p):
    values = sorted([x for x in series if not pd.isna(x)])
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[int(f)] * (c - k) + values[int(c)] * (k - f)
```

### Statistical Measures Computed

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Count** | Number of non-null values | Detect missing data |
| **Mean** | μ = Σx / n | Central tendency |
| **Std** | σ = √(Σ(x-μ)² / n) | Data spread |
| **Min** | Smallest value | Lower bound |
| **25%** | First quartile | Lower distribution |
| **50%** | Median | Middle value |
| **75%** | Third quartile | Upper distribution |
| **Max** | Largest value | Upper bound |

### What This Told Me

```
       Arithmancy      Astronomy     Herbology     ...
Count  1566.000000   1568.000000  1567.000000  ...
Mean  49634.570243     -2.975532    -5.212545  ...
Std   16679.806036    260.289446     5.219126  ...
Min  -24370.000000   -966.740000   -10.295663  ...
25%   38511.500000   -489.551387    -8.776811  ...
50%   49013.500000     -2.755013    -5.259095  ...
75%   60811.250000    489.386684    -1.645047  ...
Max  104956.000000   1016.210000    11.612895  ...
```

- Features have missing values (Count < 1600) → need imputation
- Scales are vastly different → **must normalize before training**
- Some medians ≈ means → roughly normal distributions
- Astronomy: symmetric around 0, Arithmancy: large positive values

## 📊 Data Visualization

### 1. Histogram Analysis - histogram.py

**Purpose**: Find which courses show homogeneous score distribution across houses

**My approach**: Plot each course's score distribution split by house, look for overlapping bell curves.

**Key finding**: Care of Magical Creatures shows the most homogeneous distribution → scores are similar regardless of house

### 2. Scatter Plot Analysis - scatter_plot.py

**Purpose**: Find two features that are highly correlated (similar information → redundant)

**My approach**: Plot every pair of features, look for linear patterns.

**Key finding**: **Astronomy and Defense Against the Dark Arts** show a strong linear correlation → using both would be redundant

### 3. Pair Plot - pair_plot.py

**Purpose**: Get a complete visual overview of all feature relationships at once

**What it shows**:
- **Diagonal**: Histogram of each feature per house
- **Off-diagonal**: Scatter plot of every feature pair, colored by house
- Immediately reveals which features cluster by house and which are correlated

## 🔍 Feature Analysis & Selection

After visualization, I selected features based on:

1. **Low correlation with other features**: Avoid redundant information
2. **Good house separation**: Houses have different score distributions
3. **Completeness**: Minimal missing values
4. **Variance**: Scores that differ meaningfully across students

The 7 features selected from that analysis are stored in [`theta.csv`](theta.csv):

```csv
house,bias,Herbology,Ancient Runes,Flying,Defense Against the Dark Arts,Divination,Charms,History of Magic
Gryffindor,-20.81,  -0.213,  0.009,  0.013, -0.081,  0.157, -0.055, -0.217
Hufflepuff,-13.39,   0.293, -0.019, -0.005, -0.405,  0.218, -0.079,  0.139
...
```

**Why these 7?**
- Astronomy dropped → highly correlated with DADA (redundant)
- Arithmancy, Transfiguration, Potions dropped → poor house separation
- Muggle Studies dropped → high correlation with other features

## 🎯 The Algorithm - Logistic Regression

### Conceptual Understanding

Logistic regression is like learning house preferences by weight:

```
Start: all weights = 0 (no preferences)
Loop:
  Make predictions with current weights
  Measure how wrong those predictions are
  Nudge weights in the direction that reduces error
  Repeat until predictions stabilize
```

### The Update Rules

My implementation updates parameters using these formulas:

```
For each house h, for each feature j:

  θⱼ := θⱼ - α × (1/m) × Σᵢ₌₁ᵐ (σ(z^(i)) - y^(i)) × x^(i)_j

  Where:
  - α = learning rate (step size)
  - m = number of examples
  - σ(z^(i)) = sigmoid(θ₀ + θ₁x₁ + ... + θₙxₙ)
  - y^(i) = actual label (1 if house h, 0 otherwise)
  - x^(i)_j = feature j of example i
```

**Interesting**: Identical form to linear regression's update, but `σ(z)` replaces `z` as the prediction.

### My Hyperparameters

```python
learning_rate = 0.1    # Step size for gradient descent
epochs = 1000          # Number of training iterations
```

**Why these values?**
- **α = 0.1**: Large enough to converge quickly, small enough to not overshoot
- **1000 epochs**: Enough for all 4 classifiers to converge

### Algorithm Flow

```
Initialize:
├─> tmp_theta = zeros(n_features + 1)  for each house
└─> loss_list = []

For each epoch (1000 times):
  For each house (One-vs-All):
  ├─> Compute predictions
  │   ├─> z = X_norm @ theta
  │   └─> h = sigmoid(z)
  │
  ├─> Compute gradients
  │   └─> gradient = (1/m) × X_norm.T @ (h - y_binary)
  │
  ├─> Update parameters
  │   └─> theta -= learning_rate × gradient
  │
  └─> Record loss
      └─> loss_list.append(current_loss)
```

## 💻 My Implementation Details

### Training Script - logreg_train.py

#### 1. Data Loading and Validation

```python
import pandas as pd
import numpy as np

raw_data = pd.read_csv('datasets/dataset_train.csv')

if raw_data.empty:
    print("Error: dataset_train.csv is empty or has invalid format.")
    exit(1)
```

#### 2. Preprocessing and Normalization

```python
# Impute NaN with column mean — save for prediction time
feature_means = {}
for feature in features:
    col_mean = raw_data[feature].mean()
    feature_means[feature] = col_mean
    raw_data[feature] = raw_data[feature].fillna(col_mean)

# Save means to mean.csv (needed in logreg_predict.py)
pd.DataFrame([feature_means]).to_csv("mean.csv", index=False)

# Z-score normalization
X = raw_data[features].values
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Add bias column (column of ones for θ₀)
X_norm = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
```

**Why save means to `mean.csv`?**  
Test data may have NaN values too. We must fill them with **training means**, not test means — otherwise we'd leak test information.

**Why normalize?**
- Arithmancy (~50000) vs Herbology (~-5): without normalization, gradient takes forever
- All features contribute equally to the gradient
- Faster and more stable convergence

#### 3. Initialize Training Variables

```python
houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

# One weight vector per house
n_features = X_norm.shape[1]   # features + bias
weights = {house: np.zeros(n_features) for house in houses}

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Track loss
loss_list = []
```

#### 4. Define Helper Functions

```python
def sigmoid(z):
    """Convert linear score to probability"""
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, theta):
    """Log loss (cross-entropy)"""
    h = sigmoid(X @ theta)
    return -np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
```

**Why these functions?**
- `sigmoid()`: Core of logistic regression, maps scores to probabilities
- `compute_loss()`: Implements cross-entropy exactly as defined mathematically
- `1e-10` inside log: Prevents `log(0) = -∞` which would break training

#### 5. Training Loop - The Core Algorithm

```python
for house in houses:
    theta = weights[house]

    # Binary labels: 1 if this house, 0 otherwise
    y_binary = (raw_data['Hogwarts House'] == house).astype(int).values

    for epoch in range(epochs):
        # Forward pass
        h = sigmoid(X_norm @ theta)

        # Compute gradient
        gradient = (1 / len(y_binary)) * (X_norm.T @ (h - y_binary))

        # Update weights
        theta -= learning_rate * gradient

        # Track loss
        loss_list.append(compute_loss(X_norm, y_binary, theta))

    weights[house] = theta
```

**What happens each iteration:**
1. **Forward pass**: `sigmoid(X @ θ)` → probabilities between 0 and 1
2. **Error**: `h - y` → how far off each prediction is
3. **Gradient**: `X.T @ error / m` → direction to nudge weights
4. **Update**: `θ -= α × gradient` → take a small step to reduce error

#### 6. Save Trained Model

```python
# Save all weights + feature names to theta.csv
rows = []
for house, theta in weights.items():
    row = {'house': house, 'bias': theta[0]}
    for i, feature in enumerate(features):
        row[feature] = theta[i + 1]
    rows.append(row)

pd.DataFrame(rows).to_csv("theta.csv", index=False)
print("Training complete! Weights saved to theta.csv")
```

### Prediction Script - logreg_predict.py

#### 1. Load Saved Model

```python
theta_data = pd.read_csv("theta.csv")
mean_data  = pd.read_csv("mean.csv")

features = theta_data.columns[2:].tolist()   # skip 'house' and 'bias'
houses   = theta_data['house'].tolist()
```

#### 2. Load and Preprocess Test Data

```python
raw_data = pd.read_csv(filename)

# Validate all expected features exist
for feature in features:
    if feature not in raw_data.columns:
        print(f"Error: Feature '{feature}' not found in {filename}")
        exit(1)

# Fill NaN with training means
for feature in features:
    raw_data[feature] = raw_data[feature].fillna(mean_data[feature][0])
```

**Critical**: Must use training statistics on test data — never test statistics.

#### 3. Predict

```python
# Normalize with training mean/std
X_test      = raw_data[features].values
X_test_norm = (X_test - X_mean_train) / X_std_train
X_test_norm = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])

# Compute probabilities for each house
probabilities = {}
for _, row in theta_data.iterrows():
    house = row['house']
    theta = row[1:].values.astype(float)
    probabilities[house] = sigmoid(X_test_norm @ theta)

# Pick house with highest probability
predictions = []
for i in range(len(indices)):
    probs = {house: probabilities[house][i] for house in houses}
    predictions.append(max(probs, key=probs.get))

# Save to houses.csv
pd.DataFrame({'Index': indices, 'Hogwarts House': predictions}).to_csv('houses.csv', index=False)
```

## 🎓 Training Process

### Complete Workflow

```
1. Data Loading
   ├─> Read datasets/dataset_train.csv (1600 students)
   ├─> Validate format and columns
   └─> Extract features and house labels

2. Preprocessing
   ├─> Compute feature means (for NaN imputation)
   ├─> Fill NaN values with column means
   └─> Save means → mean.csv

3. Normalization
   ├─> X_mean = mean per feature
   ├─> X_std  = std per feature
   └─> X_norm = (X - X_mean) / X_std

4. Initialize
   ├─> weights = zeros per house (4 × 8)
   │             [bias, Herbology, Ancient Runes, Flying,
   │              DADA, Divination, Charms, History of Magic]
   ├─> learning_rate = 0.1
   └─> epochs = 1000

5. Training Loop (One-vs-All, 4 houses × 1000 epochs)
   For each house:
   ├─> y_binary = (house == this_house) ? 1 : 0
   │
   └─> For each epoch:
       ├─> Forward Pass
       │   ├─> z = X_norm @ theta
       │   └─> h = sigmoid(z)
       │
       ├─> Compute Gradient
       │   └─> gradient = (1/m) × X_norm.T @ (h - y_binary)
       │
       ├─> Update Weights
       │   └─> theta -= learning_rate × gradient
       │
       └─> Track Loss
           └─> loss_list.append(cross_entropy)

6. Save Model
   ├─> theta.csv  (one row per house: bias + 7 feature weights)
   └─> mean.csv   (feature means for NaN imputation at predict time)
```

### Visualization Generation

My code creates one loss curve per house:

```python
for house in houses:
    plot.plot(range(len(loss_history[house])), loss_history[house], label=house)

plot.xlabel("Epoch")
plot.ylabel("Loss (Cross-Entropy)")
plot.title("Loss vs Epochs")
plot.legend()
plot.grid(True)
plot.savefig("loss_curve.png")
plot.close()
```

Shows convergence: cost starts at ~0.693 (random guessing = 50%) and drops steadily toward convergence.

## 🔮 Prediction System

### How a Prediction Works

**Example student** (normalized scores):
```
Herbology: 0.82, DADA: -1.23, Ancient Runes: 1.45, ...
```

**Step 1** - Compute linear score per house:
```
z_Gryffindor = bias + w₁×0.82 + w₂×(-1.23) + ...  =  1.24
z_Slytherin  = bias + w₁×0.82 + w₂×(-1.23) + ...  = -0.58
z_Ravenclaw  = bias + w₁×0.82 + w₂×(-1.23) + ...  =  0.12
z_Hufflepuff = bias + w₁×0.82 + w₂×(-1.23) + ...  = -0.85
```

**Step 2** - Convert to probabilities with sigmoid:
```
P(Gryffindor) = sigmoid( 1.24) = 0.776
P(Slytherin)  = sigmoid(-0.58) = 0.359
P(Ravenclaw)  = sigmoid( 0.12) = 0.530
P(Hufflepuff) = sigmoid(-0.85) = 0.299
```

**Step 3** - Pick winner:
```
argmax([0.776, 0.359, 0.530, 0.299]) → Gryffindor ✓
```

Note: these probabilities do **not** sum to 1 (4 independent sigmoid functions, not softmax). That's fine — we only need the argmax.

## 📈 Visualization & Analysis

### 1. histogram.py - Feature Distributions

Shows each course's score distribution split by house:
- X-axis: Score value
- Y-axis: Frequency
- 4 overlapping histograms, one per house (color-coded)

**What to look for**:
- **Peaks in different places** → good separator (houses score differently)
- **Heavily overlapping peaks** → poor separator

**Key finding**: Care of Magical Creatures shows the most similar distributions across all four houses.

### 2. scatter_plot.py - Feature Correlations

Shows scatter plots between pairs of features:
- Each point = one student
- Color = house

**What to look for**:
- **Clear linear pattern** → features are correlated (redundant)
- **Cloud of points** → no correlation (independent)

**Key finding**: Astronomy vs Defense Against the Dark Arts shows a strong linear pattern — using both would add redundancy, not information.

### 3. pair_plot.py - Full Overview

Matrix visualization of all feature relationships:
- **Diagonal**: Histogram per feature (distribution by house)
- **Off-diagonal**: Scatter plot for every pair

**What it revealed**:
- Which features clearly separate houses visually
- Which features are correlated with each other
- Overall structure of the data before building any model

### 4. loss_curve.png - Training Progress

Shows cross-entropy loss over epochs for all 4 houses:
- X-axis: Epoch
- Y-axis: Cross-Entropy Loss
- 4 lines, one per house

**Typical progression**:
```
Epoch 0:    Loss ≈ 0.693  (random guessing, sigmoid(0) = 0.5 for everyone)
Epoch 100:  Loss ≈ 0.45   (learning main patterns)
Epoch 500:  Loss ≈ 0.38   (refinement)
Epoch 900:  Loss ≈ 0.36   (convergence)
```

A smooth decreasing curve with no oscillations means the learning rate is well chosen.

## 💻 Usage

```bash
# Step 1: Statistical analysis (no libraries needed for stats)
python describe.py datasets/dataset_train.csv

# Step 2: Visualize distributions per house
python histogram.py

# Step 3: Find correlated features
python scatter_plot.py

# Step 4: Full feature overview
python pair_plot.py

# Step 5: Train the model
python logreg_train.py
# → Generates theta.csv, mean.csv, loss_curve.png

# Step 6: Predict on test set
python logreg_predict.py datasets/dataset_test.csv
# → Generates houses.csv
```

### Files Generated After Running

After `logreg_train.py`:
- ✅ `theta.csv` — weights for all 4 classifiers + feature names
- ✅ `mean.csv` — feature means for NaN imputation at test time
- ✅ `loss_curve.png` — cross-entropy over epochs for all houses

After `logreg_predict.py`:
- ✅ `houses.csv` — predicted house for each test student

### Output Format

**theta.csv**:
```csv
house,bias,Herbology,Ancient Runes,Flying,Defense Against the Dark Arts,Divination,Charms,History of Magic
Gryffindor,-20.812,-0.213,0.009,0.013,-0.081,0.157,-0.055,-0.217
Hufflepuff,-13.388,0.293,-0.019,-0.005,-0.405,0.218,-0.079,0.139
Ravenclaw,...
Slytherin,...
```

**houses.csv**:
```csv
Index,Hogwarts House
1203,Ravenclaw
617,Ravenclaw
741,Hufflepuff
...
```

## 📊 Results & Insights

### What the Weights Tell Us

From [`theta.csv`](theta.csv), the weight magnitudes reveal what the model learned:

```
house,bias,Herbology,Ancient Runes,Flying,DADA,Divination,Charms,History of Magic
Gryffindor,-20.81, -0.213,  0.009,  0.013, -0.081,  0.157, -0.055, -0.217
Hufflepuff,-13.39,  0.293, -0.019, -0.005, -0.405,  0.218, -0.079,  0.139
```

- **High positive weight** for a house → that course strongly predicts this house
- **High negative weight** → low score in that course predicts this house
- **Weight near 0** → that course doesn't help classify this house

**Example interpretation**:
- Hufflepuff has the strongest negative weight on DADA (-0.405) → Hufflepuffs tend to score lower in Defense Against the Dark Arts
- Gryffindor has a negative weight on Herbology (-0.213) → Gryffindors are not the strongest in Herbology

### Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~98-99% |
| **Training samples** | 1600 |
| **Test samples** | 400 |
| **Selected features** | 7 (from 13) |
| **Classes** | 4 (One-vs-All) |
| **Epochs** | 1000 |
| **Learning rate** | 0.1 |

### Strengths

1. ✅ **High accuracy**: ~98-99% on unseen test data
2. ✅ **Interpretable**: Weight magnitudes explain each prediction
3. ✅ **Fast training**: Converges in 1000 epochs
4. ✅ **Robust**: Handles missing values via training mean imputation
5. ✅ **Balanced**: All 4 houses classified equally well

### Limitations

1. ⚠️ **Linear boundaries**: Cannot capture complex non-linear patterns
2. ⚠️ **Feature selection**: Relies on manual visualization analysis
3. ⚠️ **Simple imputation**: Mean replacement for NaN (could use KNN)
4. ⚠️ **No softmax**: 4 independent sigmoid functions, probabilities don't sum to 1

## 🎓 Key Learnings

### Mathematical Concepts

1. **Logistic regression**: Modeling probabilities instead of raw values
2. **Sigmoid function**: Differentiable, bounded, probabilistic interpretation
3. **Cross-entropy loss**: Correct cost function for classification (not MSE)
4. **Gradient descent**: Same update form as linear regression, different prediction
5. **One-vs-All**: Reducing multi-class to multiple binary problems
6. **Z-score normalization**: Essential for gradient-based methods with mixed scales
7. **Statistical analysis**: Computing mean, std, percentiles from scratch

### Implementation Skills

1. **NumPy**: Vectorized matrix operations (no Python loops in gradient)
2. **Pandas**: Data manipulation, missing value handling, CSV I/O
3. **Matplotlib**: Histograms, scatter plots, pair plots, loss curves
4. **Manual statistics**: Rebuilding `describe()` without pandas stats
5. **Model persistence**: Saving weights and normalization params to CSV
6. **Error handling**: Validating inputs in both train and predict scripts
7. **Code organization**: Each script has a single clear responsibility

### Machine Learning Principles

1. **Preprocessing matters more than the algorithm**: Normalization and imputation are critical
2. **Visualize before modeling**: Histograms and scatter plots guided feature selection
3. **Avoid data leakage**: Always use training statistics on test data
4. **Correlation ≠ useful**: Correlated features add noise, not information
5. **Simple can be powerful**: Logistic regression achieves ~99% with good features

### Practical Insights

1. **Learning rate tuning**: α = 0.1 worked well for normalized data
2. **Convergence monitoring**: Loss curve is essential to validate training
3. **Parameter interpretation**: Weight values have real meaning (feature importance)
4. **One-vs-All is surprisingly powerful**: No need for softmax for this problem

## 📁 Files Structure

```
dslr/
│
├── README.md                     # This comprehensive guide
│
├── datasets/
│   ├── dataset_train.csv        # Training data (1600 students)
│   │                             # Format: Index, Hogwarts House, 13 course scores
│   │
│   └── dataset_test.csv         # Test data (400 students)
│                                 # Format: Index, 13 course scores (no house label)
│
├── describe.py                   # Statistical analysis tool
│                                 # - Reimplements pandas describe() from scratch
│                                 # - Computes: count, mean, std, min,
│                                 #   25%, 50%, 75%, max
│                                 # - Uses only Python math module (no NumPy stats)
│                                 # Usage: python describe.py datasets/dataset_train.csv
│
├── histogram.py                  # Distribution visualization
│                                 # - Histograms for all 13 courses
│                                 # - Split and colored by Hogwarts House
│                                 # - Used to identify homogeneous features
│
├── scatter_plot.py               # Correlation analysis
│                                 # - Scatter plots for feature pairs
│                                 # - Identifies redundant/correlated features
│                                 # - Key finding: Astronomy ↔ DADA correlation
│
├── pair_plot.py                  # Comprehensive feature visualization
│                                 # - Full matrix: histograms on diagonal,
│                                 #   scatter plots off-diagonal
│                                 # - Color-coded by Hogwarts House
│
├── logreg_train.py               # Training script
│                                 # - Loads datasets/dataset_train.csv
│                                 # - Imputes NaN with column means
│                                 # - Z-score normalizes features
│                                 # - Trains 4 binary classifiers (One-vs-All)
│                                 # - Gradient descent (1000 epochs, α=0.1)
│                                 # - Saves theta.csv + mean.csv
│                                 # - Generates loss_curve.png
│
├── logreg_predict.py             # Prediction script
│                                 # - Loads theta.csv and mean.csv
│                                 # - Validates features exist in test file
│                                 # - Imputes NaN with training means (mean.csv)
│                                 # - Normalizes with training statistics
│                                 # - Predicts house via argmax of sigmoid scores
│                                 # - Saves houses.csv
│                                 # Usage: python logreg_predict.py datasets/dataset_test.csv
│
├── theta.csv                     # Saved model weights (generated by logreg_train.py)
│                                 # Columns: house, bias, Herbology, Ancient Runes,
│                                 #          Flying, DADA, Divination, Charms,
│                                 #          History of Magic
│                                 # One row per house (4 rows total)
│
├── mean.csv                      # Feature means from training data
│                                 # (generated by logreg_train.py)
│                                 # Used to impute NaN in test data
│
├── houses.csv                    # Prediction output (generated by logreg_predict.py)
│                                 # Format: Index, Hogwarts House
│
└── loss_curve.png                # Training visualization (generated by logreg_train.py)
                                  # Cross-entropy vs epoch for all 4 houses
```

## 🔄 Future Improvements

### Model Enhancements
- [ ] Regularization (L2) to penalize large weights
- [ ] Softmax instead of independent sigmoids (proper multi-class probabilities)
- [ ] Better NaN imputation (KNN or iterative)
- [ ] Feature engineering (interaction terms)
- [ ] Cross-validation for hyperparameter tuning

### Code Quality
- [ ] Add command-line arguments (--epochs, --lr)
- [ ] Configuration file for hyperparameters
- [ ] Unit tests for all functions
- [ ] Logging instead of print statements
- [ ] Type hints for better code clarity

### Visualization
- [ ] Interactive plots (plotly)
- [ ] Confusion matrix heatmap
- [ ] ROC curves per class
- [ ] Feature importance bar chart

## 📖 References

### Theory
- École 42 DSLR subject PDF
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Logistic Regression - ML Crash Course](https://developers.google.com/machine-learning/crash-course/logistic-regression)
- [Classification - ML Crash Course](https://developers.google.com/machine-learning/crash-course/classification)

### Implementation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)

### Mathematics
- [Cross-Entropy Loss - ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
- [Sigmoid Function](https://developers.google.com/machine-learning/glossary#sigmoid-function)
- [Gradient Descent - ML Crash Course](https://developers.google.com/machine-learning/crash-course/reducing-loss/gradient-descent)
- [Z-score Normalization](https://en.wikipedia.org/wiki/Standard_score)

---

**Project Status**: ✅ Completed

**École 42 Project** | **My Grade**: 125

**Key Achievement**: Built a working multi-class logistic regression from scratch with ~98-99% accuracy!