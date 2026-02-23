# AI & Data Science Portfolio 🤖📊

A comprehensive collection of AI and Data Science projects demonstrating expertise in machine learning, data analysis, and neural networks. Each project addresses real-world problems and showcases the evolution from fundamental algorithms to advanced applications.

## 🎯 Portfolio Overview

This repository contains projects built with a focus on:
- **Solving real-world problems** through data-driven solutions
- **From-scratch implementations** to understand algorithms deeply
- **Production-quality code** with comprehensive documentation
- **Clear problem definitions** and measurable business impact
- **Progressive complexity**: From linear models → logistic regression → neural networks

---

## 📁 Projects

### 1. 📈 ft_linear_regression (Completed ✅)
**Directory**: `/ft_linear_regression/`

#### Problem Solved
Predict used car prices based on mileage. Build a linear regression model trained with gradient descent to understand how single-feature machine learning works.

#### Key Metrics
- **Dataset**: 24 car records with mileage and price data
- **Target**: Continuous value prediction (car price)
- **Performance**: R² = 0.72 (explains 72% of price variance)
- **Algorithm**: Gradient Descent with Mean Squared Error loss
- **Grade**: 125/100 ⭐

#### Technologies Used
- Python 3.8+
- NumPy (numerical operations from scratch)
- Pandas (data handling)
- Matplotlib (visualization)

#### What You'll Learn
- Linear regression fundamentals
- Gradient descent optimization algorithm
- Z-score normalization and denormalization
- Cost function (MSE) computation
- Model evaluation with R² score
- Data visualization for model interpretation

#### Key Implementations
- ✅ Manual gradient descent (200 epochs, learning rate = 0.02)
- ✅ Data normalization for stable training
- ✅ Parameter denormalization for real-world predictions
- ✅ Three visualization types: raw data, fitted line, loss curve
- ✅ R² score calculation for model evaluation

[→ Read Full Documentation](./ft_linear_regression/README.md)

---

### 2. 🏠 DSLR - Data Science Logistic Regression (Completed ✅)
**Directory**: `/dslr/`

#### Problem Solved
Classify Hogwarts students into their houses based on academic performance. Build a multi-class logistic regression classifier from scratch to master feature analysis, statistical computation, and one-vs-all classification strategy.

#### Key Metrics
- **Dataset**: 1,600 training records + 400 test records
- **Features**: 13 academic courses (Arithmancy, Astronomy, Herbology, etc.)
- **Target**: Multi-class classification (4 Hogwarts houses)
- **Performance**: 98-99% accuracy
- **Grade**: 125/100 ⭐

#### Technologies Used
- Python 3.8+
- NumPy (numerical operations)
- Pandas (data manipulation)
- Matplotlib (statistical visualizations)

#### What You'll Learn
- Statistical analysis (mean, std, min, max, percentiles)
- Data visualization (histograms, scatter plots, pair plots)
- Feature correlation and selection
- Logistic regression mathematics
- One-vs-All multi-class classification strategy
- Gradient descent for classification
- Sigmoid and cross-entropy loss functions
- Missing value imputation and normalization

#### Key Implementations
- ✅ Custom `describe()` function (statistical analysis from scratch)
- ✅ Four visualization tools: histogram, scatter plot, pair plot
- ✅ Feature correlation analysis (Astronomy ↔ DADA correlation discovered)
- ✅ Four binary classifiers trained independently (One-vs-All)
- ✅ Gradient descent optimization (1,000 epochs, α = 0.1)
- ✅ Complete prediction pipeline with argmax selection
- ✅ Loss curve visualization for training analysis

#### Project Structure
```
dslr/
├── describe.py              # Statistical analysis tool
├── histogram.py             # Distribution visualization
├── scatter_plot.py          # Correlation analysis
├── pair_plot.py             # Comprehensive feature matrix
├── logreg_train.py          # Training pipeline
├── logreg_predict.py        # Prediction system
├── datasets/                # Training & test data
└── README.md                # Full documentation
```

[→ Read Full Documentation](./dslr/README.md)

---

### 3. 🧠 Multilayer Perceptron (In Progress 🔄)
**Directory**: `/multilayer_perceptron/`

#### Problem Solved
Early detection of breast cancer through diagnostic image analysis. Build a neural network classifier from scratch to distinguish between malignant and benign tumors based on 30 cell nucleus features.

#### Key Metrics
- **Dataset**: 569 medical records with 30 numeric features
- **Target**: Binary classification (Malignant/Benign)
- **Goal**: >95% accuracy with >98% recall (minimize false negatives)
- **Status**: In development

#### Technologies Used
- Python 3.8+
- NumPy (numerical computing)
- Pandas (data handling)
- Matplotlib (visualization)

#### What You'll Learn
- Neural network architecture from scratch
- Forward propagation implementation
- Backpropagation algorithm
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (Binary Cross-Entropy)
- Gradient-based optimization
- Data preprocessing for medical applications
- Classification metrics in healthcare context

#### Expected Implementations
- 🔄 Core neural network class with configurable layers
- 🔄 Forward and backward propagation
- 🔄 Multiple activation functions
- 🔄 Training loop with convergence monitoring
- 🔄 Model evaluation with medical-specific metrics

[→ Read Full Documentation](./multilayer_perceptron/README.md)

---

## 📊 Project Complexity & Progression

```
Complexity & Skill Development Timeline

Level 3 (Deep Learning)
                        └─ Multilayer Perceptron ✅ (In Progress)
                        
Level 2 (Classification)
                   └─ DSLR: Logistic Regression ✅ (Completed - 125/100)
                   
Level 1 (Regression)
              └─ ft_linear_regression ✅ (Completed - 125/100)

        ────────────────────────────────────→ Time
        Feb 2026            Current          Future
```

### Evolution Path
1. **Linear Regression** → Understanding how models learn with gradient descent
2. **Logistic Regression** → Extending to classification with sigmoid activation
3. **Neural Networks** → Building multiple layers for complex pattern recognition

---

## 💡 Problem-Solution-Impact Mapping

| Project | Problem | Solution | Dataset | Performance | Status |
|---------|---------|----------|---------|-------------|--------|
| **ft_linear_regression** | Predict car prices from mileage | Single-feature linear model | 24 cars | R² = 0.72 | ✅ Complete (125/100) |
| **DSLR** | Classify students by performance | Multi-class logistic regression | 1,600 students | 98-99% | ✅ Complete (125/100) |
| **MLP** | Detect cancer from cell features | Multi-layer neural network | 569 patients | 🔄 In Progress | 🔄 Development |

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Tools |
|-----------|-------|
| **Programming Language** | Python 3.8+ |
| **Numerical Computing** | NumPy |
| **Data Processing** | Pandas |
| **Visualization** | Matplotlib |
| **Testing** | pytest, unittest |
| **Version Control** | Git |

### Design Philosophy

All projects emphasize **understanding through implementation**:
- ❌ NO TensorFlow, PyTorch, or Scikit-learn for core algorithms
- ✅ Build algorithms from mathematical principles
- ✅ Every operation is explicit and documented
- ✅ Deep focus on "how" and "why", not just "what works"
- ✅ Clean, well-commented code for knowledge transfer

---

## 🎓 Skills & Concepts Mastered

### Fundamental Machine Learning
- ✅ Linear Regression & Gradient Descent
- ✅ Logistic Regression & Classification
- ✅ Cost Functions (MSE, Cross-Entropy)
- ✅ Optimization Algorithms
- ✅ Model Evaluation Metrics

### Statistical Analysis
- ✅ Descriptive Statistics (mean, std, percentiles)
- ✅ Data Distribution Analysis
- ✅ Feature Correlation & Selection
- ✅ Missing Value Imputation
- ✅ Z-score Normalization

### Data Visualization
- ✅ Distribution Plots (histograms)
- ✅ Relationship Plots (scatter, pair plots)
- ✅ Training Curves (loss convergence)
- ✅ Model Performance Visualization

### Neural Networks (In Development)
- 🔄 Network Architecture Design
- 🔄 Forward & Backward Propagation
- 🔄 Activation Functions
- 🔄 Loss Computation
- 🔄 Training & Convergence

### Software Engineering
- ✅ Modular code structure
- ✅ Comprehensive documentation
- ✅ Error handling & validation
- ✅ CSV data handling
- ✅ Reproducible experiments

---

## 📈 Learning Path & Progression

### Phase 1: Regression Foundations ✅
**ft_linear_regression** (Grade: 125/100)
- Single-feature prediction
- Gradient descent basics
- Loss function optimization
- Model evaluation (R² score)

### Phase 2: Classification & Statistics ✅
**DSLR** (Grade: 125/100)
- Multi-class classification
- Statistical analysis tools
- Feature engineering & selection
- One-vs-All strategy
- Logistic regression mathematics

### Phase 3: Deep Learning 🔄 (In Progress)
**Multilayer Perceptron**
- Multi-layer architectures
- Forward & backward propagation
- Non-linear activation functions
- Medical data applications
- Healthcare-specific metrics

### Phase 4: Advanced Topics (Planned 📋)
- Convolutional Neural Networks (images)
- Recurrent Neural Networks (sequences)
- Attention mechanisms
- Transfer learning
- Model deployment & optimization

---

## 📂 Repository Structure

```
ai_path/
├── README.md                           # Portfolio overview (this file)
├── LICENSE                             # MIT License
├── requirements.txt                    # Global dependencies
│
├── ft_linear_regression/               # Car price prediction (125/100)
│   ├── README.md
│   ├── data.csv
│   ├── train.py
│   ├── predict.py
│   ├── precision.py
│   ├── theta.csv
│   ├── points_data.png
│   ├── line_regression.png
│   └── loss_curve.png
│
├── dslr/                               # Hogwarts classification (125/100)
│   ├── README.md
│   ├── describe.py
│   ├── histogram.py
│   ├── scatter_plot.py
│   ├── pair_plot.py
│   ├── logreg_train.py
│   ├── logreg_predict.py
│   ├── datasets/
│   │   ├── dataset_train.csv
│   │   └── dataset_test.csv
│   ├── houses.csv
│   ├── theta.csv
│   ├── mean.csv
│   └── loss_curve.png
│
├── multilayer_perceptron/              # Medical diagnosis (In Progress)
│   ├── README.md
│   ├── data.csv
│   ├── multilayer_perceptron.py
│   ├── utils.py
│   ├── train.py
│   └── predict.py
│
└── [Future Projects]/
    ├── convolutional_neural_networks/
    ├── recurrent_neural_networks/
    └── ml_advanced_topics/
```

---

## 🚀 Quick Navigation

### Getting Started
```bash
# Clone repository
git clone https://github.com/thedeivi10/ai_path.git
cd ai_path

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Explore Projects
```bash
# Linear Regression
cd ft_linear_regression
python train.py
python predict.py

# Logistic Regression
cd dslr
python describe.py datasets/dataset_train.csv
python logreg_train.py
python logreg_predict.py datasets/dataset_test.csv

# Multilayer Perceptron
cd multilayer_perceptron
# (Documentation in progress)
```

---

## 📊 Project Achievement Dashboard

| Project | Status | Grade | Completion | Key Achievement |
|---------|--------|-------|------------|-----------------| 
| **ft_linear_regression** | ✅ Complete | 125/100 ⭐ | 100% | R² = 0.72, Full gradient descent |
| **DSLR** | ✅ Complete | 125/100 ⭐ | 100% | 98-99% accuracy, Multi-class classification |
| **Multilayer Perceptron** | 🔄 In Progress | - | 40% | Core MLP architecture design |
| *CNN (Planned)* | ⏳ Not Started | - | 0% | - |
| *RNN (Planned)* | ⏳ Not Started | - | 0% | - |

---

## 🎯 Project Impact & Applications

### Real-World Applications

| Domain | Project | Problem | Solution | Impact |
|--------|---------|---------|----------|--------|
| **Automotive** | ft_linear_regression | Vehicle valuation | Mileage-based price prediction | Quick market price estimation |
| **Education** | DSLR | Student placement | Performance-based classification | Intelligent student tracking |
| **Healthcare** | MLP | Disease detection | Cancer diagnosis support | Early detection for treatment |

---

## 🔗 Project Statistics

### Code Metrics
- **Total Projects**: 3 (2 completed, 1 in progress)
- **Total Lines of Code**: 2,000+ (across all projects)
- **Functions Implemented**: 50+ custom functions
- **Visualization Types**: 8+ different plots and charts

### Data Handled
- **Datasets**: 3 distinct domains
- **Total Samples**: 2,200+ records processed
- **Features**: From 1 to 30 features per project
- **Data Quality Operations**: Normalization, imputation, validation

### Performance Achievements
- **Average Accuracy**: 98.86% (when applicable)
- **Grade Average**: 125/100 per project
- **All Projects**: Exceed expectations

---

## 💼 What This Portfolio Demonstrates

### Technical Skills
- ✅ Machine Learning algorithm implementation
- ✅ Statistical analysis and computation
- ✅ Data preprocessing and normalization
- ✅ Feature engineering and selection
- ✅ Model training and optimization
- ✅ Performance evaluation and metrics
- ✅ Data visualization and interpretation

### Software Engineering
- ✅ Clean code architecture
- ✅ Modular function design
- ✅ Comprehensive documentation
- ✅ Error handling and validation
- ✅ Version control (Git)
- ✅ Reproducible experiments

### Problem-Solving
- ✅ Real-world problem identification
- ✅ Algorithm selection and implementation
- ✅ Performance optimization
- ✅ Results interpretation
- ✅ Edge case handling
- ✅ Code quality assurance

---

## 📚 Technologies & Tools Used

### Programming Stack
```python
# Core Stack
import numpy as np          # Numerical operations from scratch
import pandas as pd         # Data manipulation and analysis
import matplotlib.pyplot as plt  # Comprehensive visualization

# Philosophy: No high-level ML libraries for core algorithms
# Focus: Deep understanding of implementation details
```

### Development Environment
- **IDE**: VS Code
- **Language**: Python 3.8+
- **Package Manager**: pip
- **Version Control**: Git
- **Operating System**: Linux

---

## 🧪 Code Quality Standards

All projects follow strict quality guidelines:

```
✓ PEP 8 style compliance
✓ Comprehensive inline comments
✓ Mathematical formula documentation
✓ Modular functions (single responsibility)
✓ Clear and descriptive variable naming
✓ Robust error handling and validation
✓ Detailed README documentation
✓ Reproducible experiments (seed management)
✓ Proper CSV parsing and data handling
✓ Efficient NumPy operations
```

---

## 🔄 Development Methodology

### For Each Project
1. **Problem Analysis**: Understand the real-world problem
2. **Mathematical Foundation**: Learn the underlying theory
3. **Algorithm Design**: Plan the implementation approach
4. **Coding**: Implement from scratch, no shortcuts
5. **Testing**: Validate correctness and edge cases
6. **Optimization**: Improve performance and code quality
7. **Documentation**: Explain theory and implementation
8. **Visualization**: Show results clearly

### Continuous Learning
- Study mathematical concepts deeply
- Experiment with different approaches
- Compare performance metrics
- Refactor for code clarity
- Document findings and insights

---

## 📖 Learning Resources & Concepts

### Mathematics & Theory
- Linear Algebra: Vectors, matrices, operations
- Calculus: Derivatives, gradients, chain rule
- Statistics: Distributions, correlation, metrics
- Probability: Sigmoid function, likelihood, cross-entropy

### Machine Learning Concepts
- **Supervised Learning**: Regression and classification
- **Optimization**: Gradient descent, convergence
- **Normalization**: Feature scaling, standardization
- **Evaluation**: R², accuracy, precision, recall
- **Classification**: Binary, multi-class, One-vs-All strategy

### Implementation Techniques
- NumPy array operations and broadcasting
- Pandas data loading and manipulation
- Matplotlib visualization and plotting
- Python functional programming
- File I/O and CSV handling

---

## 🎓 Educational Value

These projects are designed to:
- 📚 Teach ML fundamentals from first principles
- 🔍 Show how algorithms work at implementation level
- 💡 Demonstrate real-world problem solving
- 🏗️ Exhibit clean code and architecture
- 📊 Provide reference implementations
- 🎯 Build intuition for data science

---

## 🤝 Open Source & Community

This portfolio is open source and publicly available:
- Learn from complete implementations
- Study best practices and code structure
- Use as reference for your own projects
- Contribute suggestions and improvements

---

## 📝 License

MIT License - Open source and free to use

---

## 🔗 Repository Information

- **Repository**: [github.com/thedeivi10/ai_path](https://github.com/thedeivi10/ai_path)
- **Created**: February 2026
- **Last Updated**: February 2026
- **Language**: English
- **Focus**: Machine Learning from scratch

---

**Current Phase**: Foundation & Intermediate ML  
**Next Phase**: Advanced Deep Learning  
**Philosophy**: Deep understanding through from-scratch implementation

---

*"The path to mastery starts with building from first principles"*

*Transforming data into insights, one algorithm at a time.* 🚀