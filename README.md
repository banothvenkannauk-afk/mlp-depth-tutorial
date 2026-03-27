# How Deep Should a Neural Network Be?
### Understanding Underfitting and Overfitting in Multilayer Perceptrons

**Author:** Banoth Venkanna  
**Module:** Machine Learning / Neural Networks  
**Date:** March 2026  

---

## Overview

This project is a tutorial-style university assignment that investigates how the depth of a neural network affects its performance. Three Multilayer Perceptron (MLP) architectures of increasing depth are compared on the scikit-learn Diabetes regression dataset, demonstrating underfitting, balanced fitting, and overfitting.

---

## Files

| File | Description |
|------|-------------|
| `mlp_experiment.py` | Python script — trains all 3 models and generates the 3 figures |
| `mlp_depth_tutorial.pdf` | Full assignment report (PDF) |
| `requirements.txt` | Python dependencies |

---

## Models Compared

| Model | Architecture | Expected Behaviour |
|-------|-------------|-------------------|
| Model A | (8) | Underfitting |
| Model B | (32, 16) | Balanced / Good generalisation |
| Model C | (128, 64, 32, 16) | Overfitting |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/banothvenkannauk-afk/mlp-depth-tutorial.git
cd mlp-depth-tutorial
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the experiment**
```bash
python mlp_experiment.py
```

This will train all three models and save the following figures:
- `figure1_comparison.png` — Train vs Test MSE bar chart
- `figure2_scatter.png` — Predicted vs Actual scatter plot
- `figure3_loss_curves.png` — Training loss curves over epochs

---

## Dataset

- **Name:** Diabetes dataset (scikit-learn built-in)
- **Samples:** 442
- **Features:** 10 physiological baseline measurements
- **Target:** Quantitative measure of disease progression after one year
- **Split:** 80% train / 20% test
- **Preprocessing:** StandardScaler (zero mean, unit variance)

---

## Key Findings

- A shallow network (Model A) underfit — high error on both train and test sets
- A moderate network (Model B) generalised best — lowest test MSE, small train–test gap
- A deep network (Model C) overfit — low train error but higher test error and diverging loss curves
- **Conclusion:** Bigger is not always better. Depth should be matched to the complexity of the data.

---

## Requirements

```
scikit-learn
numpy
matplotlib
```
