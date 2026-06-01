# DSLR - Data Science Logistic Regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Neural%20Networks-013243?style=for-the-badge&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![42](https://img.shields.io/badge/Ecole%2042-Artificial%20Intelligence-black?style=for-the-badge)

</div>

---

## Table of Contents

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

---

## Overview

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

---

## Mathematical Foundation

### The Logistic Model

Unlike linear regression, logistic regression models probabilities using the sigmoid function:

```
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))

where: z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| Input | Output |
|-------|--------|
| 0 | 0.5 |
| +∞ | → 1 |
| -∞ | → 0 |

**Why sigmoid?** Maps predictions to probabilities, smooth and differentiable (needed for gradient descent), and has a natural interpretation as likelihood.

### The Cost Function - Log Loss (Cross-Entropy)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Why cross-entropy and not MSE?** Convex function (guaranteed global minimum), penalizes confident wrong predictions very heavily, and is mathematically compatible with sigmoid for clean gradients.

### Gradient Descent Update Rule

$$\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left(\sigma(z^{(i)}) - y^{(i)}\right) \cdot x^{(i)}_j$$

This formula looks identical to linear regression's update rule — the key difference is that `σ(z)` replaces `z` as the prediction.

### Multi-Class Classification - One-vs-All

For 4 houses, I train 4 independent binary classifiers:

```
1. Is this student Gryffindor?   (1 = yes, 0 = no)
2. Is this student Slytherin?    (1 = yes, 0 = no)
3. Is this student Ravenclaw?    (1 = yes, 0 = no)
4. Is this student Hufflepuff?   (1 = yes, 0 = no)

Final prediction: argmax(P₁, P₂, P₃, P₄)
```

---

## Understanding the Problem

### The Dataset

| Property | Value |
|----------|-------|
| Input | 13 course scores per student |
| Output | Hogwarts House (4 classes) |
| Training samples | 1,600 students |
| Test samples | 400 students |

### Features (13 courses)

Arithmancy · Astronomy · Herbology · Defense Against the Dark Arts · Divination · Muggle Studies · Ancient Runes · History of Magic · Transfiguration · Potions · Care of Magical Creatures · Charms · Flying

### Challenges

- Missing values (NaN) scattered across features
- Very different scales (Arithmancy: ~50,000 vs Herbology: ~-5)
- 13 dimensions with potentially redundant features
- 4 classes to classify simultaneously

---

## Statistical Analysis - describe.py

I recreated pandas' `describe()` function entirely from scratch using only Python's `math` module:

```python
def mean(series):
    values = [x for x in series if not pd.isna(x)]
    return sum(values) / len(values) if values else float('nan')

def std(series):
    values = [x for x in series if not pd.isna(x)]
    m = mean(series)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def percentile(series, p):
    values = sorted([x for x in series if not pd.isna(x)])
    k = (len(values) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[int(f)] * (c - k) + values[int(c)] * (k - f)
```

### Metrics Computed

| Metric | Formula | Purpose |
|--------|---------|---------|
| Count | Non-null values | Detect missing data |
| Mean | μ = Σx / n | Central tendency |
| Std | σ = √(Σ(x-μ)² / n) | Data spread |
| Min / Max | Smallest / largest | Bounds |
| 25% / 50% / 75% | Quartiles | Distribution shape |

### What This Revealed

- Features have missing values (Count < 1600) → need imputation
- Scales differ massively → **must normalize before training**
- Astronomy is symmetric around 0; Arithmancy has large positive values

---

## Data Visualization

### histogram.py — Feature Distributions

Plots each course's score distribution split by house. Used to identify the most homogeneous feature across houses.

**Key finding**: Care of Magical Creatures shows the most similar distributions across all four houses.

### scatter_plot.py — Correlation Analysis

Scatter plots between every pair of features, colored by house. Used to identify redundant features.

**Key finding**: Astronomy and Defense Against the Dark Arts show a strong linear correlation — using both adds redundancy, not information.

### pair_plot.py — Full Overview

Full matrix visualization:
- **Diagonal**: Histogram per feature, colored by house
- **Off-diagonal**: Scatter plot for every feature pair

Immediately reveals which features cluster by house and which are correlated.

---

## Feature Analysis & Selection

After visualization, 7 features were selected based on low inter-feature correlation and clear house separation:

```
Herbology · Ancient Runes · Flying · Defense Against the Dark Arts
Divination · Charms · History of Magic
```

Features dropped and why:
- **Astronomy**: highly correlated with DADA (redundant)
- **Arithmancy, Transfiguration, Potions**: poor house separation
- **Muggle Studies**: high correlation with other features

---

## The Algorithm - Logistic Regression

### Conceptual Understanding

Start with all weights at zero. For each iteration: make predictions, measure how wrong they are, nudge weights in the direction that reduces error. Repeat until predictions stabilize.

### Hyperparameters

```python
learning_rate = 0.1    # Step size for gradient descent
epochs        = 1000   # Training iterations
```

### Algorithm Flow

```
Initialize: weights = zeros(n_features + 1) for each house

For each epoch (1000 times):
  For each house (One-vs-All):
  ├── Forward pass:     z = X_norm @ theta
  │                     h = sigmoid(z)
  ├── Compute gradient: gradient = (1/m) × X_norm.T @ (h - y_binary)
  ├── Update weights:   theta -= learning_rate × gradient
  └── Record loss:      loss_list.append(cross_entropy)
```

---

## My Implementation Details

### Preprocessing and Normalization

```python
# Impute NaN with column means — save for prediction time
for feature in features:
    col_mean = raw_data[feature].mean()
    feature_means[feature] = col_mean
    raw_data[feature] = raw_data[feature].fillna(col_mean)

# Save means to mean.csv (needed in logreg_predict.py)
pd.DataFrame([feature_means]).to_csv("mean.csv", index=False)

# Z-score normalization
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias column
X_norm = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
```

**Why save means?** Test data may have NaN values too. We must fill them with training means — not test means — to avoid data leakage.

### Core Helper Functions

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, theta):
    h = sigmoid(X @ theta)
    return -np.mean(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))
```

`1e-10` inside log prevents `log(0) = -∞` from breaking training.

### Training Loop

```python
for house in houses:
    theta    = weights[house]
    y_binary = (raw_data['Hogwarts House'] == house).astype(int).values

    for epoch in range(epochs):
        h        = sigmoid(X_norm @ theta)
        gradient = (1 / len(y_binary)) * (X_norm.T @ (h - y_binary))
        theta   -= learning_rate * gradient
        loss_list.append(compute_loss(X_norm, y_binary, theta))

    weights[house] = theta
```

### Save Trained Model

```python
rows = []
for house, theta in weights.items():
    row = {'house': house, 'bias': theta[0]}
    for i, feature in enumerate(features):
        row[feature] = theta[i + 1]
    rows.append(row)

pd.DataFrame(rows).to_csv("theta.csv", index=False)
```

---

## Training Process

### Complete Workflow

```
1. Load            datasets/dataset_train.csv (1,600 students)
2. Impute NaN      fill with column means → save to mean.csv
3. Normalize       X_norm = (X − μ) / σ
4. Add bias        prepend column of ones
5. Train           4 × 1000 epochs, one binary classifier per house
6. Save model      theta.csv (weights) + mean.csv (imputation values)
7. Generate plot   loss_curve.png (cross-entropy per house over epochs)
```

---

## Prediction System

### How a Prediction Works

For a given student (normalized scores):

```
Step 1 — Linear score per house:
  z_Gryffindor = bias + w₁×Herbology + w₂×DADA + ...  =  1.24
  z_Slytherin  = ...                                    = -0.58
  z_Ravenclaw  = ...                                    =  0.12
  z_Hufflepuff = ...                                    = -0.85

Step 2 — Convert to probabilities:
  P(Gryffindor) = sigmoid(1.24)  = 0.776
  P(Slytherin)  = sigmoid(-0.58) = 0.359
  P(Ravenclaw)  = sigmoid(0.12)  = 0.530
  P(Hufflepuff) = sigmoid(-0.85) = 0.299

Step 3 — Pick winner:
  argmax([0.776, 0.359, 0.530, 0.299]) → Gryffindor ✓
```

Note: probabilities do not sum to 1 (4 independent sigmoids, not softmax). Only the argmax matters.

### Critical: Use Training Statistics on Test Data

```python
# Fill NaN with training means — never test means
for feature in features:
    raw_data[feature] = raw_data[feature].fillna(mean_data[feature][0])

# Normalize with training mean/std
X_test_norm = (X_test - X_mean_train) / X_std_train
```

---

## Visualization & Analysis

### histogram.py — Distributions per House

Each course plotted as 4 overlapping histograms (one per house). Peaks in different places indicate good separators; heavily overlapping peaks indicate poor ones.

### scatter_plot.py — Feature Correlations

A clear linear pattern between two features means they're correlated and one can be dropped. A scattered cloud means they are independent.

### pair_plot.py — Full Feature Overview

Matrix of all feature relationships at a glance. Reveals clustering by house and cross-feature correlations before building any model.

### loss_curve.png — Training Progress

Cross-entropy for all 4 classifiers over 1,000 epochs:

```
Epoch 0:    Loss ≈ 0.693  (random guessing)
Epoch 100:  Loss ≈ 0.45   (learning main patterns)
Epoch 500:  Loss ≈ 0.38   (refinement)
Epoch 1000: Loss ≈ 0.36   (convergence)
```

A smooth decreasing curve with no oscillations confirms a well-chosen learning rate.

---

## Usage

```bash
# Step 1: Statistical analysis
python describe.py datasets/dataset_train.csv

# Step 2: Visualize distributions
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

### Generated Files

| File | Description |
|------|-------------|
| `theta.csv` | Weights for all 4 classifiers |
| `mean.csv` | Training feature means (for NaN imputation at predict time) |
| `loss_curve.png` | Cross-entropy over epochs per house |
| `houses.csv` | Predicted house for each test student |

---

## Results & Insights

### What the Weights Tell Us

```
house,       bias,   Herbology, Ancient Runes, Flying, DADA,   Divination, Charms, History of Magic
Gryffindor, -20.81,  -0.213,    0.009,         0.013, -0.081,  0.157,     -0.055, -0.217
Hufflepuff, -13.39,   0.293,   -0.019,        -0.005, -0.405,  0.218,     -0.079,  0.139
```

- **High positive weight** → that course strongly predicts this house
- **High negative weight** → low score in that course predicts this house
- **Weight near 0** → course does not help classify this house

**Example**: Hufflepuff has the strongest negative weight on DADA (-0.405), suggesting Hufflepuffs tend to score lower in Defense Against the Dark Arts.

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~98–99% |
| Training samples | 1,600 |
| Test samples | 400 |
| Selected features | 7 (from 13) |
| Classes | 4 (One-vs-All) |
| Epochs | 1,000 |
| Learning rate | 0.1 |

### Strengths

- High accuracy: ~98–99% on unseen test data
- Interpretable: weight magnitudes explain each prediction
- Robust: handles missing values via training mean imputation
- Balanced: all 4 houses classified equally well

### Limitations

- Linear decision boundaries: cannot capture complex non-linear patterns
- Manual feature selection: relies on visualization analysis
- Simple imputation: mean replacement for NaN (KNN would be more accurate)
- Independent sigmoids: probabilities per house do not sum to 1

---

## Key Learnings

- **Preprocessing matters more than the algorithm**: normalization and imputation are critical
- **Visualize before modeling**: histograms and scatter plots guided feature selection
- **Avoid data leakage**: always use training statistics on test data
- **Correlation ≠ useful**: correlated features add noise, not information
- **Simple can be powerful**: logistic regression achieves ~99% with well-chosen features
- **One-vs-All is effective**: no need for softmax for this problem

---

## Files Structure

```
dslr/
│
├── README.md                   # This guide
│
├── datasets/
│   ├── dataset_train.csv       # Training data (1,600 students, 13 courses + house)
│   └── dataset_test.csv        # Test data (400 students, 13 courses, no house label)
│
├── describe.py                 # Statistical analysis — reimplements pandas describe()
│                               # from scratch using only Python's math module
│
├── histogram.py                # Distribution per course, split by house
│                               # Used to identify homogeneous features
│
├── scatter_plot.py             # Feature correlation analysis
│                               # Key finding: Astronomy ↔ DADA
│
├── pair_plot.py                # Full feature matrix (histograms + scatter plots)
│                               # colored by Hogwarts House
│
├── logreg_train.py             # Training script
│                               # Imputes NaN, normalizes, trains 4 classifiers
│                               # (One-vs-All, 1000 epochs, α=0.1)
│                               # Saves theta.csv, mean.csv, loss_curve.png
│
├── logreg_predict.py           # Prediction script
│                               # Loads theta.csv + mean.csv, normalizes test data,
│                               # predicts house via argmax(sigmoid scores)
│                               # Saves houses.csv
│
├── theta.csv                   # Saved model weights (4 rows: one per house)
│                               # Columns: house, bias, 7 feature weights
│
├── mean.csv                    # Training feature means for NaN imputation
│
├── houses.csv                  # Prediction output (Index, Hogwarts House)
│
└── loss_curve.png              # Cross-entropy vs epoch for all 4 houses
```

---

## Technologies

Python · NumPy · Pandas · Matplotlib · Machine Learning · Logistic Regression · Gradient Descent

---

## Author

**David Gómez-Landero López**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/david-gomez-landero)