import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

try:
    from A7 import fit, predict, score
    print("Custom kNN package imported successfully.")
except ImportError:
    raise ImportError(
        "Please ensure knn_package.py (from A7) is in the same directory."
    )

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
    'vote_tie_break': 'nearest',
    'weights': 'uniform'
}

plot_file = "knn_comparison_plot.png"

def load_splits():
    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(YTR_CSV).squeeze()
    Xte  = pd.read_csv(XTE_CSV)
    yte  = pd.read_csv(YTE_CSV).squeeze()
    return Xtr, ytr, Xte, yte

def run_experiment(Xtr, ytr, Xte, yte, k_values):
    my_acc = []
    sk_acc = []

    for k in k_values:

        params = dict(params_base)
        params['n_neighbors'] = k
        m = fit(Xtr, ytr, params)
        my_a = score(m, Xte, yte)
        my_acc.append(my_a)

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(Xtr, ytr)
        sk_a = neigh.score(Xte, yte)
        sk_acc.append(sk_a)

        print(f"k = {k:2d} | Custom Acc: {my_a:.4f} | Sklearn Acc: {sk_a:.4f}")

    return my_acc, sk_acc

def plot_results(k_values, my_acc, sk_acc):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, my_acc, marker='o', label='Custom kNN', linewidth=2)
    plt.plot(k_values, sk_acc, marker='s', label='Scikit-learn kNN', linewidth=2)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('kNN Accuracy Comparison: Custom vs Scikit-learn')
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

    my_acc, sk_acc = run_experiment(Xtr, ytr, Xte, yte, ks)
    plot_results(ks, my_acc, sk_acc)

    res = pd.DataFrame({
        'k': ks,
        'Custom_Accuracy': my_acc,
        'Sklearn_Accuracy': sk_acc
    })
    print("\nSummary Table:")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
