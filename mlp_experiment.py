# =============================================================================
# mlp_experiment.py
# How Deep Should a Neural Network Be?
# Understanding Underfitting and Overfitting in Multilayer Perceptrons
#
# Author : Banoth Venkanna
# Module : Machine Learning / Neural Networks
# Date   : March 2026
# Github : https://github.com/banothvenkannauk-afk/mlp-depth-tutorial.git
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ── 1. Load and prepare the dataset ──────────────────────────────────────────

print("Loading Diabetes dataset...")
X, y = load_diabetes(return_X_y=True)

# 80/20 train-test split with fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardise features (zero mean, unit variance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")

# ── 2. Define the three architectures ────────────────────────────────────────

models = {
    "Model A — Underfitting  (8)":            (8,),
    "Model B — Balanced      (32, 16)":       (32, 16),
    "Model C — Overfitting   (128, 64, 32, 16)": (128, 64, 32, 16),
}

EPOCHS      = 100
RANDOM_SEED = 42

trained   = {}   # fitted MLPRegressor objects
train_mse = {}   # final train MSE per model
test_mse  = {}   # final test  MSE per model
loss_curves = {}  # loss curve (one value per epoch) per model

# ── 3. Train each model ───────────────────────────────────────────────────────

print("\nTraining models...")
for name, hidden_layers in models.items():
    print(f"  {name} ...", end=" ", flush=True)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=1,              # we step manually to collect loss per epoch
        warm_start=True,
        random_state=RANDOM_SEED,
        learning_rate_init=0.001,
    )

    curve = []
    for epoch in range(EPOCHS):
        mlp.fit(X_train, y_train)
        curve.append(mlp.loss_)   # training loss after this epoch

    trained[name]     = mlp
    loss_curves[name] = curve
    train_mse[name]   = mean_squared_error(y_train, mlp.predict(X_train))
    test_mse[name]    = mean_squared_error(y_test,  mlp.predict(X_test))

    print(f"Train MSE = {train_mse[name]:.1f}  |  Test MSE = {test_mse[name]:.1f}")

# ── 4. Figure 1 — Model Comparison Bar Chart ─────────────────────────────────

def plot_comparison():
    labels     = ['Model A\n(8)', 'Model B\n(32, 16)', 'Model C\n(128,64,32,16)']
    train_vals = [train_mse[k] for k in models]
    test_vals  = [test_mse[k]  for k in models]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, train_vals, w, label='Train MSE', color='#1a4d6e', zorder=3)
    b2 = ax.bar(x + w/2, test_vals,  w, label='Test MSE',  color='#b8d0df', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean Squared Error', fontsize=9)
    ax.set_title('Figure 1 — Train vs Test MSE by Architecture', fontsize=10, pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.grid(axis='y', color='#d6cfc7', linewidth=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=7.5)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    plt.savefig('figure1_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: figure1_comparison.png")

# ── 5. Figure 2 — Prediction vs Actual Scatter ───────────────────────────────

def plot_scatter():
    colours = {'Model A': '#e07b54', 'Model B': '#1a4d6e', 'Model C': '#6b9e78'}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot([50, 320], [50, 320], '--', color='#aaa', linewidth=1.2,
            label='Ideal (y = x)', zorder=1)

    for (name, mlp), colour, short in zip(
        trained.items(),
        ['#e07b54', '#1a4d6e', '#6b9e78'],
        ['Model A', 'Model B', 'Model C']
    ):
        preds = mlp.predict(X_test)
        ax.scatter(y_test, preds, alpha=0.7, color=colour, s=30,
                   label=short, zorder=3)

    ax.set_xlabel('Actual value', fontsize=9)
    ax.set_ylabel('Predicted value', fontsize=9)
    ax.set_title('Figure 2 — Predicted vs Actual (Test Set)', fontsize=10, pad=10)
    ax.set_xlim(40, 330); ax.set_ylim(40, 330)
    ax.grid(color='#d6cfc7', linewidth=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figure2_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: figure2_scatter.png")

# ── 6. Figure 3 — Training Loss Curves ───────────────────────────────────────

def plot_loss_curves():
    colours = ['#e07b54', '#1a4d6e', '#6b9e78']
    labels  = ['Model A', 'Model B', 'Model C']
    epochs  = range(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for (name, curve), colour, label in zip(loss_curves.items(), colours, labels):
        ax.plot(epochs, curve, color=colour, linewidth=2.0, label=label)

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Training Loss (MSE)', fontsize=9)
    ax.set_title('Figure 3 — Training Loss Curves', fontsize=10, pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.grid(color='#d6cfc7', linewidth=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('figure3_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: figure3_loss_curves.png")

# ── 7. Print summary table ────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*60)
    print(f"{'Model':<45} {'Train MSE':>10} {'Test MSE':>10}")
    print("-"*60)
    for name in models:
        print(f"{name:<45} {train_mse[name]:>10.1f} {test_mse[name]:>10.1f}")
    print("="*60)

# ── 8. Run everything ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating figures...")
    plot_comparison()
    plot_scatter()
    plot_loss_curves()
    print_summary()
    print("\nDone! All figures saved.")
