# ft_linear_regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Neural%20Networks-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![42](https://img.shields.io/badge/Ecole%2042-Artificial%20Intelligence-black?style=for-the-badge)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Understanding the Problem](#understanding-the-problem)
- [Data Normalization - Z-Score Implementation](#data-normalization---z-score-implementation)
- [The Algorithm - Gradient Descent](#the-algorithm---gradient-descent)
- [My Implementation Details](#my-implementation-details)
- [Denormalization Process](#denormalization-process)
- [Training Process](#training-process)
- [Prediction System](#prediction-system)
- [Model Evaluation - R² Score](#model-evaluation---r²-score)
- [Visualization & Analysis](#visualization--analysis)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Key Learnings](#key-learnings)
- [Files Structure](#files-structure)

---

## Overview

**ft_linear_regression** is my first machine learning project at École 42. The goal is to implement a linear regression model from scratch to predict car prices based on mileage, using gradient descent optimization.

**The Challenge**:
- Build everything from scratch using only Python, NumPy, and Matplotlib
- No scikit-learn or high-level ML libraries
- Understand every line of math and code

**What I Built**:
1. A training system that learns parameters θ₀ and θ₁
2. A prediction system that estimates car prices
3. An evaluation system that measures model accuracy (R²)
4. Visualizations to understand the data and results

---

## Mathematical Foundation

### The Linear Model

```
ŷ = θ₀ + θ₁ × x
```

| Symbol | Meaning |
|--------|---------|
| **ŷ** | Predicted value (estimated price) |
| **θ₀** | Intercept (base price) |
| **θ₁** | Slope (price change per km) |
| **x** | Input feature (mileage) |

### The Cost Function - Mean Squared Error

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

- **m** = number of training examples (24)
- **ŷᵢ** = predicted price for example i
- **yᵢ** = actual price for example i
- **1/2** simplifies the derivative

**Why squared error?** Makes all errors positive, penalizes large errors more heavily, and creates a smooth convex function that is easy to optimize.

---

## Understanding the Problem

### Dataset — data.csv

24 car sales with mileage and price:

```csv
km,price
240000,3650
139800,3800
150500,4400
...
```

| Property | Value |
|----------|-------|
| Samples | 24 |
| Mileage range | 22,899 – 240,000 km |
| Price range | 3,650 – 8,290 € |
| Correlation | Negative (higher mileage → lower price) |

### The Challenge: Different Scales

Raw data has very different magnitudes — mileage values like 240,000 dominate gradient calculations, causing slow or unstable training. **Solution**: Z-score normalization.

---

## Data Normalization - Z-Score Implementation

### The Formula

$$z = \frac{x - \mu}{\sigma}$$

Transforms data to have **mean = 0** and **standard deviation = 1**.

### My Implementation

```python
# Step 1: Calculate mean
mileage_mean = sum(mileage) / len(mileage)

# Step 2: Calculate standard deviation
variance = sum((x - mileage_mean) ** 2 for x in mileage) / len(mileage)
mileage_std = math.sqrt(variance)

# Step 3: Normalize
mileage_norm = (mileage - mileage_mean) / mileage_std
```

### Before and After

```
Before:  mileage = [240000, 139800, 150500, ...]   ← Huge numbers
After:   mileage = [  2.64,   0.58,   0.80, ...]   ← Consistent scale
```

**Note**: Only mileage (X) is normalized, not price (y). This simplifies denormalization.

### Benefits

- Faster convergence: gradient descent reaches optimum quicker
- Numerical stability: prevents overflow/underflow
- Better learning rate: same α works across different scales

---

## The Algorithm - Gradient Descent

### Conceptual Understanding

Gradient descent is like finding the valley while blindfolded: feel which direction is downhill (compute gradient), take a step (update parameters), repeat until you reach the bottom (convergence).

### The Update Rules

$$\theta_0 := \theta_0 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

$$\theta_1 := \theta_1 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i$$

### Hyperparameters

```python
epochs        = 200   # Training iterations
learning_rate = 0.02  # Step size (α)
```

### Algorithm Flow

```
Initialize: tmp_theta0 = 0, tmp_theta1 = 0

For each epoch (200 times):
├── Calculate predictions:  ŷᵢ = tmp_theta0 + tmp_theta1 × x_norm[i]
├── Compute gradients:      sum0 = Σ(ŷᵢ - yᵢ),  sum1 = Σ(ŷᵢ - yᵢ) × xᵢ
├── Update parameters:      tmp_theta0 -= α × (sum0 / m)
│                           tmp_theta1 -= α × (sum1 / m)
└── Record loss:            loss_list.append(MSE)
```

---

## My Implementation Details

### Training Script — train.py

#### Data Loading and Validation

```python
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
except (OSError, IOError):
    print("Error: data.csv is empty or has invalid format.")
    exit(1)

if np.any(data < 0) or np.any(data > 1000000):
    print("Error: All values must be between 0 and 1,000,000.")
    exit(1)
```

#### Core Helper Functions

```python
def predict(mileage):
    return tmp_theta0 + tmp_theta1 * mileage

def compute_loss():
    total = 0.0
    for i in range(len(mileage_norm)):
        error = predict(mileage_norm[i]) - price[i]
        total += error * error
    return total / (2 * len(mileage_norm))
```

#### Training Loop

```python
for _ in range(epochs):
    sum0 = 0
    sum1 = 0
    for i in range(len(mileage_norm)):
        error = predict(mileage_norm[i]) - price[i]
        sum0 += error
        sum1 += error * mileage_norm[i]
    tmp_theta0 -= learning_rate * (1 / len(mileage_norm)) * sum0
    tmp_theta1 -= learning_rate * (1 / len(mileage_norm)) * sum1
    loss_list.append(compute_loss())
```

---

## Denormalization Process

After training, parameters work on **normalized** mileage but users provide **real** km values. The solution is a two-step algebraic transformation.

### Formulas

```python
theta1 = tmp_theta1 / mileage_std
theta0 = tmp_theta0 - theta1 * mileage_mean
```

### Mathematical Derivation

Starting from the normalized model and substituting `mileage_norm = (mileage - μ) / σ`:

```
price = tmp_theta0 + tmp_theta1 × [(mileage - μ) / σ]
      = [tmp_theta0 - (tmp_theta1/σ) × μ] + [tmp_theta1/σ] × mileage
           └──────────────────┘               └──────────┘
                  theta0                         theta1
```

The saved parameters in `theta.csv` work directly on real km — no normalization needed at prediction time.

---

## Training Process

### Complete Workflow

```
1. Load & validate     data.csv
2. Normalize           mileage_norm = (mileage - μ) / σ
3. Initialize          tmp_theta0 = 0, tmp_theta1 = 0
4. Train (200 epochs)  gradient descent update per epoch
5. Denormalize         theta1 = tmp_theta1 / σ
                       theta0 = tmp_theta0 − theta1 × μ
6. Save model          theta.csv
7. Generate plots      points_data.png, line_regression.png, loss_curve.png
```

---

## Prediction System

### predict.py

```python
def estimatePrice(mileage):
    return theta0 + theta1 * mileage
```

- Graceful fallback: if model is not trained, uses θ₀=0, θ₁=0
- Input validation: checks for negative values
- Formatted output to 2 decimal places

### Usage Example

```bash
$ python predict.py
Enter mileage (km): 150000
Estimated price: 5189.61 €

$ python predict.py
Enter mileage (km): 50000
Estimated price: 7296.61 €
```

---

## Model Evaluation - R² Score

### precision.py

$$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

| R² Value | Meaning |
|----------|---------|
| 1.0 | Perfect predictions |
| **0.72** | **My model — 72% variance explained** |
| 0.0 | No better than predicting the mean |
| < 0.0 | Worse than predicting the mean |

28% of variance is explained by other factors (age, brand, condition) not present in this single-feature model.

---

## Visualization & Analysis

### points_data.png — Raw Data

Scatter plot of the 24 data points showing the clear downward trend (negative correlation).

### line_regression.png — Fitted Model

Trained line overlaid on data:

```
price = 8350.11 − 0.02107 × mileage

Interpretation:
- Base price at 0 km:       8,350 €
- Price decrease per 1 km:  0.021 €
- Price decrease per 10k km: ~211 €
```

### loss_curve.png — Training Progress

MSE decreasing over 200 epochs:

```
High  |*
Loss  | **
      |   ***
      |      *****_______
Low   |                  ────── ← Convergence
      +─────────────────────────
      0       100       200
                Epochs
```

A smooth decreasing curve confirms a well-chosen learning rate and successful convergence.

---

## Usage

```bash
# Step 1: Train the model
python train.py
# → Saves theta.csv, generates 3 plots

# Step 2: Make a prediction
python predict.py
# Enter mileage (km): 120000
# Estimated price: 5822.71 €

# Step 3: Evaluate accuracy
python precision.py
# The coefficient of determination is: 0.7234
# This means the model explains 72.34% of the variance.
```

### Generated Files

| File | Description |
|------|-------------|
| `theta.csv` | Trained model parameters |
| `points_data.png` | Raw data scatter plot |
| `line_regression.png` | Fitted regression line |
| `loss_curve.png` | MSE over training epochs |

---

## Results & Insights

### Final Model Parameters

```
theta0 =  8350.109  €      (y-intercept)
theta1 = -0.02107   €/km   (slope)
```

### Predictions at Key Mileages

| Mileage (km) | Estimated Price (€) |
|---|---|
| 50,000 | ~7,295 |
| 100,000 | ~6,243 |
| 150,000 | ~5,190 |
| 200,000 | ~4,138 |

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.72 |
| Training samples | 24 |
| Features | 1 (mileage) |
| Epochs | 200 |
| Learning rate | 0.02 |

### Strengths

- Simple and interpretable: easy to understand and explain
- Good R²: 72% is solid for a single-feature model
- Fast training: converges in 200 epochs
- Stable: no oscillations or divergence

### Limitations

- Single feature: real prices depend on age, brand, condition, and more
- Small dataset: only 24 samples limits generalization
- Linear assumption: relationship might benefit from polynomial features
- Extrapolation risk: predictions outside training range (22k–240k km) are unreliable

---

## Key Learnings

- **Preprocessing matters**: normalization dramatically improves training stability
- **Visualize everything**: plots reveal insights no numbers alone can show
- **Denormalization is essential**: parameters must be converted back to real scale
- **Simple can be powerful**: linear regression achieves strong results with clean data
- **Build from scratch**: implementing math manually deepens understanding far beyond using libraries

---

## Files Structure

```
ft_linear_regression/
│
├── README.md               # This guide
├── data.csv                # Training dataset (24 samples: km, price)
│
├── train.py                # Training script
│                           # Loads, validates, normalizes, trains,
│                           # denormalizes, saves theta.csv, generates plots
│
├── predict.py              # Prediction interface
│                           # Loads theta.csv, takes mileage input,
│                           # returns estimated price
│
├── precision.py            # Model evaluation
│                           # Calculates and prints R² score
│
├── theta.csv               # Trained model parameters (theta0, theta1)
│
├── points_data.png         # Raw data scatter plot
├── line_regression.png     # Fitted model visualization
└── loss_curve.png          # MSE vs epoch (training progress)
```

---

## Technologies

Python · NumPy · Matplotlib · Machine Learning · Linear Regression · Gradient Descent

---

## Author

**David Gómez-Landero López**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/david-gomez-landero)
