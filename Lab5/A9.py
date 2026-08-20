import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

try:
    from A7 import fit, predict, score
    print("Custom kNN package imported successfully.")
except ImportError:
    raise ImportError("Please ensure knn_package.py (from A7) is in the same directory.")

X_TRAIN_CSV = "X_train.csv"
YTR_CSV = "y_train.csv"
XTE_CSV  = "X_test.csv"
YTE_CSV  = "y_test.csv"

ks = list(range(1, 16))

params_base = {
    'metric': 'euclidean',
    'sort_algorithm': 'quick',
    'numeric_imputation': 'mean',
    'categorical_imputation': 'mode',
    'categorical_encoding': 'onehot',
    'vote_tie_break': 'nearest'
}

plot_file = "knn_weighted_comparison_plot.png"

def load_splits():
    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(YTR_CSV).squeeze()
    Xte  = pd.read_csv(XTE_CSV)
    yte  = pd.read_csv(YTE_CSV).squeeze()
    return Xtr, ytr, Xte, yte

def run_experiment(Xtr, ytr, Xte, yte, k_values):
    u_acc = []
    d_acc = []
    sk_u_acc = []

    for k in k_values:

        p_u = dict(params_base)
        p_u['n_neighbors'] = k
        p_u['weights'] = 'uniform'
        m_u = fit(Xtr, ytr, p_u)
        acc_custom_uniform = score(m_u, Xte, yte)
        u_acc.append(acc_custom_uniform)

        p_d = dict(params_base)
        p_d['n_neighbors'] = k
        p_d['weights'] = 'distance'
        m_d = fit(Xtr, ytr, p_d)
        acc_custom_distance = score(m_d, Xte, yte)
        d_acc.append(acc_custom_distance)

        neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        neigh.fit(Xtr, ytr)
        sk_a = neigh.score(Xte, yte)
        sk_u_acc.append(sk_a)

        print(f"k = {k:2d} | Custom Uniform: {acc_custom_uniform:.4f} | "
              f"Custom Distance: {acc_custom_distance:.4f} | "
              f"Sklearn Uniform: {sk_a:.4f}")

    return u_acc, d_acc, sk_u_acc

def plot_results(k_values, u_acc, d_acc, sk_acc):
    plt.figure(figsize=(12, 7))
    plt.plot(k_values, u_acc, marker='o', label='Custom kNN (uniform)', linewidth=2)
    plt.plot(k_values, d_acc, marker='^', label='Custom kNN (distance)', linewidth=2)
    plt.plot(k_values, sk_acc, marker='s', label='Scikit-learn kNN (uniform)', linewidth=2)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('kNN Accuracy Comparison: Custom Uniform vs Custom Weighted vs Scikit-learn')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.show()
    print(f"\nPlot saved as '{plot_file}'")

def main():
    Xtr, ytr, Xte, yte = load_splits()
    print(f"Training samples: {Xtr.shape[0]}")
    print(f"Test samples    : {Xte.shape[0]}\n")

    u_acc, d_acc, sk_acc = run_experiment(
        Xtr, ytr, Xte, yte, ks
    )
    plot_results(ks, u_acc, d_acc, sk_acc)

    res = pd.DataFrame({
        'k': ks,
        'Custom_Uniform': u_acc,
        'Custom_Distance': d_acc,
        'Sklearn_Uniform': sk_acc
    })
    print("\nSummary Table:")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
