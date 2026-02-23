# Multilayer Perceptron (In Progress)

## 🎯 Problem Statement

Early detection of breast cancer is crucial for patient survival rates. This project implements a neural network from scratch to classify breast cancer tumors as **malignant or benign** based on diagnostic features extracted from digitized cell nucleus images.

**The Challenge**: Given 30 features from cell nuclei measurements, build a model that can accurately predict whether a tumor is malignant (dangerous) or benign (non-cancerous) with high precision to minimize false negatives.

## 📋 Project Overview

This project involves implementing a multilayer perceptron neural network entirely from scratch to solve a binary classification problem. The focus is on understanding the mathematical foundations of neural networks: forward propagation, backpropagation, and gradient-based optimization.

## 📊 Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

- **Purpose**: Classify tumors as Malignant (M) or Benign (B)
- **Samples**: ~569 medical records
- **Features**: 30 continuous numeric values derived from cell nucleus images
- **Classes**: Binary (2 classes)

### Feature Categories
Features include measurements like:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Symmetry, fractal dimension
- Statistical variations (mean, standard error, worst)

### Example Data Format
```
ID,Diagnosis,Feature1,Feature2,...,Feature30
8712766,M,17.47,24.68,116.1,984.6,0.1049,...,0.093
89382602,B,12.76,13.37,82.29,504.1,0.08794,...,0.08253
```

## 🧠 What This Project Teaches

### Core Concepts Implemented

1. **Neural Network Architecture**
   - Multi-layer perceptron design
   - Configurable hidden layers and neurons
   - Different activation functions

2. **Forward Propagation**
   - Matrix operations for data flow
   - Activation function applications
   - Output prediction generation

3. **Backpropagation Algorithm**
   - Gradient computation using chain rule
   - Error flow backward through layers
   - Weight and bias updates

4. **Optimization**
   - Stochastic Gradient Descent (SGD)
   - Learning rate effects
   - Convergence monitoring

5. **Data Preprocessing**
   - Feature normalization
   - Train-test splitting
   - Handling data imbalance

## 🏗️ Project Structure

```
multilayer_perceptron/
├── README.md                    # Project documentation
├── data.csv                     # Breast cancer dataset
├── multilayer_perceptron.py     # Core MLP implementation
├── utils.py                     # Helper functions
├── train.py                     # Training pipeline
└── predict.py                   # Inference script
```

## 🎓 Learning Outcomes

By the end of this project, you will understand:

- ✅ How neural networks learn through backpropagation
- ✅ The mathematical foundations of deep learning
- ✅ How to normalize and preprocess medical data
- ✅ Trade-offs between model complexity and generalization
- ✅ How to evaluate classification models in healthcare context
- ✅ Why high recall is critical in medical diagnosis (minimize false negatives)

## 🚀 Implementation Status

### Phase 1: Foundation (Current)
- [x] Data loading and exploration
- [ ] Data preprocessing and normalization
- [ ] Neural network architecture design

### Phase 2: Core Algorithm
- [ ] Forward propagation implementation
- [ ] Activation functions (ReLU, Sigmoid, Tanh)
- [ ] Loss function (Binary Cross-Entropy)

### Phase 3: Training
- [ ] Backpropagation algorithm
- [ ] Gradient descent optimization
- [ ] Training loop with convergence

### Phase 4: Evaluation
- [ ] Model evaluation metrics (Accuracy, Precision, Recall, F1)
- [ ] Confusion matrix analysis
- [ ] Hyperparameter tuning

### Phase 5: Documentation
- [ ] Code documentation
- [ ] Mathematical explanations
- [ ] Usage examples

## 🔧 Architecture Overview

```
Input Layer
(30 features from cell measurements)
        ↓
Hidden Layer 1 (e.g., 64 neurons, ReLU)
        ↓
Hidden Layer 2 (e.g., 32 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)
        ↓
Binary Classification: Malignant or Benign
```

## 📈 Expected Outcomes

- **Classification Accuracy**: Target >95% on test set
- **Recall (Critical for medical)**: >98% (minimize false negatives)
- **Precision**: >92% (minimize false positives)
- **Model Convergence**: Smooth loss decrease during training

## 🎯 Business Impact

- **Medical Application**: Support doctors in early cancer detection
- **False Negative Cost**: Minimized through high recall optimization
- **Interpretability**: Understanding feature importance for diagnosis
- **Scalability**: Model can be deployed for screening programs

---

**Project Type**: Binary Classification with Neural Networks
**Domain**: Medical Diagnosis / Healthcare AI
**Status**: In Progress 🔄
**Last Updated**: February 2026