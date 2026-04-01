# MNIST Handwritten Digit Recognition - Neural Network from Scratch

A systematic exploration of feedforward neural networks applied to the classic MNIST handwritten digit recognition problem. This project builds 14 progressively complex models — starting from zero hidden layers and raw gradient descent, all the way to deep 3-layer networks — to understand **how architecture, batch size, activation functions, and training duration each affect performance**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Approach Overview](#approach-overview)
- [Models & Experiments](#models--experiments)
- [Activation Functions Explained](#activation-functions-explained)
- [Results Summary](#results-summary)
- [Overfitting Analysis](#overfitting-analysis)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)

---

## Problem Statement

**Given a 28×28 grayscale image of a handwritten digit (0–9), predict which digit it is.**

This is a 10-class image classification problem. The goal is not just to build a classifier, but to *understand* what makes a neural network learn well — by comparing models that differ in exactly one or two variables at a time.

### Why MNIST?

MNIST is the "Hello World" of machine learning. It is:
- Small enough to train quickly on a laptop
- Complex enough to reward good architectures
- Well-understood, so results are easy to benchmark

---

## Dataset

**Source:** [MNIST](http://yann.lecun.com/exdb/mnist/) — loaded directly via `keras.datasets.mnist`

| Split | Samples |
|-------|---------|
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |
| **Total** | **70,000** |

Each image is 28×28 pixels. Each pixel is an integer from 0 (black) to 255 (white).

### Preprocessing Steps

1. **Normalize pixel values**: Divide all pixel values by 255 → range becomes [0, 1].  
   *Why? Neural networks train faster and more stably when inputs are small numbers near zero.*

2. **Flatten images**: Reshape each 28×28 image into a flat vector of 784 numbers.  
   *Why? Dense (fully connected) layers expect 1D input.*

3. **One-hot encode labels**: Convert digit labels (e.g., `7`) into a 10-element binary vector (e.g., `[0,0,0,0,0,0,0,1,0,0]`).  
   *Why? Categorical cross-entropy loss requires probability-style targets.*

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| TensorFlow | 2.18.0 | Deep learning framework |
| Keras | (bundled) | High-level model building API |
| NumPy | 1.26.4 | Numerical operations |
| Pandas | 2.2.2 | Results tabulation |
| Matplotlib | 3.8.3 | Plotting loss/accuracy curves |
| Seaborn | 0.13.2 | Visualizations |
| scikit-learn | 1.3.2 | Train/val splitting |

---

## Project Structure

```
MNIST_NN/
├── MNIST_NeuralNetwork.ipynb   # Main notebook with all 14 models
└── README.md
```

---

## Approach Overview

The project follows a **controlled experiment** strategy. Each model changes **one or two variables** from the previous, so the effect of each change is clearly visible. The progression is:

```
No hidden layers
    → Vary epochs (10 → 50)
    → Vary batch size (full batch → 32 → 64)

One hidden layer
    → Try full batch vs small batch
    → Vary neurons (64 → 128)
    → Try Sigmoid → Tanh → ReLU

Two hidden layers
    → Mix activation functions (ReLU→Tanh vs Tanh→ReLU)

Three hidden layers
    → Vary epochs (50 → 100)
```

All models use:
- **Optimizer**: SGD (Stochastic Gradient Descent) with default learning rate (~0.01)
- **Loss function**: Categorical cross-entropy
- **Output layer**: Dense(10, activation='softmax')
- **Metric**: Accuracy

---

## Models & Experiments

### Baseline: No Hidden Layers

A neural network with no hidden layers is essentially a **logistic regression** — it can only learn linear decision boundaries.

| Model | Epochs | Batch Size | Val Accuracy | Train Time |
|-------|--------|------------|--------------|------------|
| 0 | 10 | 50,000 (full) | 16.3% | 5s |
| 1 | 50 | 50,000 (full) | 43.3% | 20s |
| 2 | 10 | 32 | 90.7% | 45s |
| 3 | 50 | 32 | 91.9% | 221s |
| 4 | 50 | 64 | 91.4% | 113s |

**What we learned:**
- Full-batch gradient descent (one weight update per epoch) learns extremely slowly.
- Switching to mini-batch SGD (batch=32 → ~1,563 updates per epoch) jumps accuracy from 43% to 91% in the same number of epochs.
- Smaller batch = more weight updates = faster, better learning.

---

### One Hidden Layer

Adding a hidden layer gives the network the ability to learn **non-linear** patterns.

```
Input (784) → Dense(neurons, activation) → Dense(10, softmax)
```

| Model | Neurons | Activation | Batch | Val Accuracy |
|-------|---------|------------|-------|--------------|
| 5 | 64 | Sigmoid | 50,000 | 15.4% |
| 6 | 64 | Sigmoid | 32 | 94.1% |
| 7 | 128 | Sigmoid | 32 | 93.9% |
| 8 | 128 | Tanh | 32 | 96.7% |
| 9 | 128 | ReLU | 32 | **97.2%** |

**What we learned:**
- Sigmoid with full-batch is nearly useless (15%). Even with small batches, it tops out ~94%.
- Tanh is significantly better than Sigmoid — jumps to 96.7%.
- ReLU is the winner in a single-layer setup — 97.2% validation accuracy.
- Doubling neurons from 64 to 128 did *not* help (Sigmoid actually got slightly worse). Architecture matters more than raw size.

---

### Two Hidden Layers

```
Input (784) → Dense(128, act1) → Dense(64, act2) → Dense(10, softmax)
```

| Model | Layer 1 | Layer 2 | Val Accuracy | Train Accuracy |
|-------|---------|---------|--------------|----------------|
| 10 | ReLU | Tanh | **97.5%** | 99.7% |
| 11 | Tanh | ReLU | 97.5% | 99.4% |

**What we learned:**
- Mixed activation functions (ReLU + Tanh) marginally outperform a single activation.
- Both configurations reach ~97.5% validation accuracy — the order of activations makes little practical difference.
- A growing gap between train (99.7%) and validation (97.5%) signals the beginning of overfitting.

---

### Three Hidden Layers

```
Input (784) → Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(10, softmax)
```

| Model | Epochs | Val Accuracy | Train Accuracy | Val Loss |
|-------|--------|--------------|----------------|----------|
| 12 | 50 | 97.4% | **99.98%** | 0.119 |
| 13 | 100 | 97.4% | **100.0%** | 0.136 |

**What we learned:**
- Adding a 3rd hidden layer causes severe overfitting: train accuracy hits 100% while validation accuracy stays at 97.4%.
- Doubling the epochs from 50 to 100 provides zero improvement on validation — the model is memorizing training data, not generalizing.
- Validation *loss* actually increased with more epochs (0.119 → 0.136) — a textbook overfitting signature.

---

## Activation Functions Explained

| Function | Formula | Range | Best Use | Issue |
|----------|---------|-------|----------|-------|
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output | Vanishing gradients in deep networks |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Better than sigmoid in hidden layers | Still has vanishing gradient issue |
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers in most modern networks | Dead neurons at x<0 |
| **Softmax** | eˣⁱ/Σeˣʲ | (0,1), sums to 1 | Multi-class output layer | — |

**In plain English:**
- Sigmoid squashes everything to [0,1] — good for binary output but problematic deep in the network because gradients become tiny (vanishing gradient problem).
- Tanh is like a centered sigmoid — gradients are better but still vanish for large inputs.
- ReLU simply passes positive values through unchanged. Fast to compute, avoids vanishing gradients in forward layers, and consistently outperforms both above in our experiments.

---

## Results Summary

All 14 models ranked by validation accuracy:

| Rank | Model | Hidden Layers | Architecture | Activations | Batch | Epochs | Val Acc | Train Acc |
|------|-------|---------------|--------------|-------------|-------|--------|---------|-----------|
| 1 | 10 | 2 | [128, 64] | ReLU → Tanh | 32 | 50 | **97.50%** | 99.73% |
| 2 | 11 | 2 | [128, 64] | Tanh → ReLU | 32 | 50 | 97.46% | 99.44% |
| 3 | 13 | 3 | [128, 64, 32] | ReLU → ReLU → ReLU | 32 | 100 | 97.45% | 100.00% |
| 4 | 12 | 3 | [128, 64, 32] | ReLU → ReLU → ReLU | 32 | 50 | 97.44% | 99.98% |
| 5 | 9 | 1 | [128] | ReLU | 32 | 50 | 97.18% | 98.62% |
| 6 | 8 | 1 | [128] | Tanh | 32 | 50 | 96.70% | 98.07% |
| 7 | 6 | 1 | [64] | Sigmoid | 32 | 50 | 94.13% | 94.56% |
| 8 | 7 | 1 | [128] | Sigmoid | 32 | 50 | 93.91% | 94.26% |
| 9 | 3 | 0 | — | — | 32 | 50 | 91.86% | 92.39% |
| 10 | 4 | 0 | — | — | 64 | 50 | 91.44% | 91.86% |
| 11 | 2 | 0 | — | — | 32 | 10 | 90.71% | 90.83% |
| 12 | 1 | 0 | — | — | 50000 | 50 | 43.32% | 43.06% |
| 13 | 0 | 0 | — | — | 50000 | 10 | 16.28% | 15.27% |
| 14 | 5 | 1 | [64] | Sigmoid | 50000 | 50 | 15.41% | 14.81% |

**Best model: Model 10** — 2 hidden layers [128 ReLU → 64 Tanh], batch 32, 50 epochs — 97.50% validation accuracy.

---

## Overfitting Analysis

Overfitting happens when a model performs well on training data but poorly on new, unseen data. It learns the training examples by "memory" rather than learning generalizable patterns.

### How We Detected It

By plotting training accuracy vs validation accuracy across epochs:

- **No overfitting** (Models 0–4): Both curves track closely. The model is *underfitting* — too simple to learn the data.
- **Mild overfitting** (Models 9–11): ~2% gap between train (98–99%) and validation (97%). Acceptable.
- **Severe overfitting** (Models 12–13): Train hits 99.98–100%, validation plateaus at 97.4%. Validation loss increases while training loss decreases — the model memorizes training data.

### What Causes It

Deeper networks have more parameters, giving them more "memory capacity." Without regularization, they use that capacity to memorize rather than generalize.

### What We Did NOT Apply (Future Work)

The current experiments explore architectural choices only. The notebook identifies these as next steps to reduce overfitting in the 3-layer models:

| Technique | What it does |
|-----------|-------------|
| **Dropout** | Randomly turns off neurons during training — forces redundant representations |
| **L2 Regularization** | Penalizes large weights — discourages over-reliance on any feature |
| **Batch Normalization** | Normalizes layer inputs — stabilizes training and acts as mild regularizer |
| **Early Stopping** | Stops training when validation loss stops improving |
| **Learning Rate Scheduling** | Reduces LR over time — prevents overshooting the optimal point |

---

## Key Findings

### 1. Batch Size Is the Most Impactful Single Change

| Change | Accuracy Gain |
|--------|--------------|
| Full batch (50,000) → Batch 32 | +74% (16% → 90%) |
| 10 epochs → 50 epochs | +1.2% |
| 0 hidden layers → 1 hidden layer (ReLU) | +5.3% |
| 1 hidden layer → 2 hidden layers | +0.3% |

**Mini-batch SGD (batch=32) is ~1,563× more updates per epoch than full-batch GD.** More frequent weight updates means the model learns far faster and reaches a better optimum.

### 2. ReLU Consistently Outperforms Sigmoid and Tanh

For hidden layers:
- Sigmoid: ~94% (suffers from vanishing gradient problem)
- Tanh: ~96.7% (better gradient flow)
- ReLU: ~97.2% (best — no saturation for positive inputs)

### 3. More Depth ≠ More Accuracy Without Regularization

| Depth | Val Accuracy | Overfitting |
|-------|-------------|-------------|
| 0 layers | 91.9% | None (underfitting) |
| 1 layer | 97.2% | Mild |
| 2 layers | 97.5% | Moderate |
| 3 layers | 97.4% | Severe |

The sweet spot is **2 hidden layers**. A 3rd layer adds no validation accuracy improvement but nearly memorizes the training set entirely.

### 4. More Epochs Don't Help Beyond a Point

Model 12 (50 epochs) and Model 13 (100 epochs) have identical validation accuracy (~97.4%). The extra 50 epochs only worsened overfitting. The model converges around epoch 30–40.

### 5. Activation Function Order in Mixed-Layer Networks

Swapping the activation order (ReLU→Tanh vs Tanh→ReLU) makes virtually no difference in this setup (~0.04% delta). The type of activations matters far more than the order.

---

## Visualizations in the Notebook

1. **Sample digit grid** — 2 sample images for each digit (0–9), 10×2 grid
2. **Digit frequency bar chart** — distribution of classes in training and test sets
3. **Training vs Validation Loss curves** — plotted per model across all epochs
4. **Training vs Validation Accuracy curves** — per model, showing convergence and overfitting patterns
5. **Model comparison table** — all 14 models with hyperparameters and results side by side

---

## How to Run

### Requirements

```bash
pip install tensorflow==2.18.0 scikit-learn matplotlib seaborn numpy pandas
```

### Run the Notebook

```bash
jupyter notebook MNIST_NeuralNetwork.ipynb
```

Or open in VS Code with the Jupyter extension, or upload to [Google Colab](https://colab.research.google.com/).

### Run All Cells

The notebook is self-contained. Running all cells in order will:
1. Download and preprocess the MNIST dataset
2. Train all 14 models sequentially
3. Print training/validation metrics per epoch
4. Generate all comparison plots
5. Display the final results summary table

> **Note:** Full training takes approximately 35–40 minutes on a CPU due to the 14 sequential model runs.

---

## Concepts for Beginners

If you are new to machine learning, here is a quick glossary of terms used throughout this project:

| Term | Plain English |
|------|--------------|
| **Neural Network** | A system of layers of interconnected "neurons" that learns patterns from examples |
| **Layer** | A group of neurons. Input layer receives data, hidden layers transform it, output layer produces predictions |
| **Neuron** | A single unit that takes multiple inputs, multiplies by learned weights, adds a bias, and applies an activation function |
| **Activation Function** | A mathematical function that adds non-linearity — without it, layers could only learn straight lines |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch Size** | How many training examples are used per weight update. Smaller = more updates per epoch |
| **Loss** | A number measuring how wrong the model is. We minimize this during training |
| **Gradient Descent** | The algorithm that adjusts weights by moving in the direction that reduces loss |
| **Overfitting** | When a model performs great on training data but poorly on new data — it memorized instead of learned |
| **Dropout** | A regularization technique that randomly disables neurons during training to prevent overfitting |
| **Softmax** | An activation function for the output layer that converts raw scores into probabilities summing to 1 |

---

## Keywords

`machine-learning` `deep-learning` `neural-network` `mnist` `image-classification` `tensorflow` `keras` `python` `jupyter-notebook` `sgd` `relu` `sigmoid` `tanh` `softmax` `overfitting` `regularization` `hyperparameter-tuning` `classification` `digit-recognition` `feedforward-network`
