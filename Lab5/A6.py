import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_TRAIN_CSV = "X_train.csv"
YTR_CSV = "y_train.csv"
XTE_CSV  = "X_test.csv"
YTE_CSV  = "y_test.csv"

k = 3

show_n = 10

def load_splits():
    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(YTR_CSV).squeeze()
    Xte  = pd.read_csv(XTE_CSV)
    yte  = pd.read_csv(YTE_CSV).squeeze()
    return Xtr, ytr, Xte, yte

def main():

    Xtr, ytr, Xte, yte = load_splits()
    print(f"Training samples: {Xtr.shape[0]}")
    print(f"Test samples    : {Xte.shape[0]}\n")

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtr, ytr)

    pred = neigh.predict(Xte)

    print(f"First {show_n} preds (true label vs predicted):")
    print("Index\tTrue\tPredicted")
    for i in range(min(show_n, len(yte))):
        true = yte.iloc[i]
        guess = pred[i]
        marker = "  ✓" if true == guess else "  ✗"
        print(f"{i}\t{true}\t{guess}{marker}")
    print()

    correct = np.sum(pred == yte)
    wrong = len(yte) - correct
    print(f"Correct preds  : {correct} / {len(yte)}")
    print(f"Incorrect preds: {wrong} / {len(yte)}\n")

    acc = accuracy_score(yte, pred)
    print(f"Overall Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(yte, pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(yte, pred)
    print(cm)
    print()

    print("Misclassified samples (index, true, predicted):")
    bad = np.where(pred != yte)[0]
    for idx in bad[:10]:
        print(f"  {idx}: true={yte.iloc[idx]}, predicted={pred[idx]}")
    if len(bad) == 0:
        print("  None")
    else:
        print(f"  ... total {len(bad)} misclassified samples")

if __name__ == "__main__":
    main()
