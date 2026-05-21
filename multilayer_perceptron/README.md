```markdown
# 🧠 Multilayer Perceptron (MLP) from Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Science-013243.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-150458.svg)
![Ecole 42](https://img.shields.io/badge/Ecole_42-Project-000000.svg)

## 🎯 Project Overview
This project is an advanced, fully functional implementation of a feedforward **Multilayer Perceptron (Neural Network) built entirely from scratch**, developed as part of the Ecole 42 Data Science curriculum. 

The core objective of this project is to "open the black box" of deep learning. Rather than relying on high-level frameworks like TensorFlow, PyTorch, or Scikit-Learn, this project manually implements the dense mathematical foundations of neural networks, including **Vectorized Forward Propagation**, **Chain-Rule Backpropagation**, and **Gradient Descent Optimization** using purely NumPy.

The model is purpose-built for Binary Classification, specifically analyzing tabular data tracking cell properties to diagnose breast cancer tumors as Malignant ('M') or Benign ('B').

---

## 🔬 Technical Implementation & Features

This MLP goes beyond the standard curriculum requirements by implementing several advanced Machine Learning techniques and bonuses:

* **Pure Mathematical implementation:** Core operations (Matrix dot products, Activation derivatives) are computed entirely through NumPy broadcasting for optimal performance.
* **Categorical Cross-Entropy & Softmax:** Implements the mathematically robust Softmax activation alongside Cross-Entropy loss for probability distribution outputs.
* **Z-Score Normalization:** Automatically standardizes input features (mean=0, std=1) to prevent vanishing or exploding gradients.
* **Adam Optimizer:** Upgrades standard Gradient Descent with Adaptive Moment Estimation (Adam) for vastly superior convergence rates.
* **Mini-Batch Training:** Dynamically partitions the dataset to update network weights iteratively, minimizing memory overhead and assisting the network in escaping local minima.
* **Early Stopping Mechanism:** Prevents overfitting by monitoring the validation loss and dynamically halting training after a patience limit is breached.
* **Comprehensive Validation Metrics:** Continuously tracks advanced evaluation metrics throughout training, including **Accuracy, Precision, Recall, and F1-Score**.
* **Visual Learning Curves:** Automatically generates `.png` graphs plotting Training vs. Validation Loss and Accuracy side-by-side using `matplotlib` to compare multiple learning rates inside the same batch of tests.
* **Logging:** Fully traceable metrics histories mapped to dedicated CSV files per epoch.

---

## 🛠️ Setup & Installation

To run the Multilayer Perceptron, you must have Python 3 installed. We highly recommend using a virtual environment to isolate the project's dependencies.

**1. Clone the repository and navigate to the project directory:**
```bash
git clone https://github.com/your-username/multilayer_perceptron.git
cd multilayer_perceptron
```

**2. Create a virtual environment:**
```bash
python3 -m venv venv
```

**3. Activate the virtual environment:**
* **Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```
* **Windows:**
  ```bash
  venv\Scripts\activate
  ```

**4. Install project dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

The program (mlp.py) operates via the command-line interface, gracefully guiding users through the three primary phases of the Machine Learning pipeline: **1. Data Splitting**, **2. Training**, and **3. Inference (Prediction)**.

### Phase 1: Splitting the Dataset
First, isolate an evaluation dataset to strictly prevent data leakage during training. This command takes your raw data and generates a mathematically safe 80/20 stratified split (`train.csv` and `test.csv`).
```bash
python mlp.py --dataset data.csv
```

### Phase 2: Training the Network
Train the network using `train.csv`. The script features an incredibly flexible CLI argument parser that allows deep customization of the network's architecture and hyperparameters.

**Flag Structure Requirements:**
`--training --layer [hidden_neurons...] --epochs [int] --loss [categoricalCrossentropy] --batch_size [int] --learning_rate [float] --optimization [GD/Adam]`

**Example A: Training a Standard Network**
Train a single network featuring two hidden layers (24 neurons each), using the Adam optimizer and a robust mini-batch size of 32 for 100 epochs.
```bash
python mlp.py --training --layer 24 24 --epochs 100 --loss categoricalCrossentropy --batch_size 32 --learning_rate 0.01 --optimization Adam
```

**Example B: Batch Training & Model Structural Comparison**
You can chain multiple network configurations together in a single command. The script will train each configuration sequentially. Within each configuration, the network will automatically test variations of the learning rate ($LR$, $LR \times 10$, $LR / 10$) allowing you to visually analyze and find the absolute optimal convergence rate efficiently.
```bash
python mlp.py --training \
  --layer 5 5 --epochs 150 --loss categoricalCrossentropy --batch_size 16 --learning_rate 0.005 --optimization GD \
  --layer 16 16 --epochs 200 --loss categoricalCrossentropy --batch_size 32 --learning_rate 0.01 --optimization Adam
```

**Training Artifacts Generated:**
* 💾 `models.npz`: A highly compressed archive saving the optimal network weights, biases, and standardizing feature constraints.
* 📊 `history_model_X_lr_Y.csv`: Step-by-step sequential logs mapping out validation trajectories.
* 📉 `curves_arch-(*).png`: Visual diagnostic graphs plotting Loss and Accuracy trajectories relative to the learning rates.

### Phase 3: Making Predictions
Deploy the optimal trained weights saved in `models.npz` directly to classify entirely unseen data. By default, it runs predictions on the generated `test.csv`.
```bash
python mlp.py --predict
```

To strictly evaluate a separate or custom `.csv` dataset, pass the file location as the third parameter:
```bash
python mlp.py --predict test.csv
```
*Outputs are completely formatted, decoded back to original 'M' vs 'B' formats, and exported locally to `results.csv`.*

---

## 💼 Why This Matters (Key Learnings)
Building an MLP entirely from scratch deliberately strips away the automated magic and abstraction of modern AI frameworks. It forces an unwavering and rigorous understanding of the Calculus powering backpropagation, the multi-dimensional Linear Algebra orchestrating dense layers, and the architectural design protocols actively required to negotiate obstacles such as vanishing gradients or severe data overfitting. This project distinctly bridges the gap between pure theoretical mathematics and practical, applicable Software Engineering.
