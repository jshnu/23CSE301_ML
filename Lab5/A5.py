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

def load_splits():
    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(YTR_CSV).squeeze()
    Xte  = pd.read_csv(XTE_CSV)
    yte  = pd.read_csv(YTE_CSV).squeeze()
    return Xtr, ytr, Xte, yte

def main():

    Xtr, ytr, Xte, yte = load_splits()

    print(f"Training set: {Xtr.shape[0]} samples")
    print(f"Test set    : {Xte.shape[0]} samples")
    print(f"Classes in test set: {yte.unique()}\n")

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtr, ytr)

    test_accuracy = neigh.score(Xte, yte)
    print(f"Test Accuracy (using neigh.score): {test_accuracy:.4f}")

    pred = neigh.predict(Xte)
    print("\nClassification Report:")
    print(classification_report(yte, pred))

    print("Confusion Matrix:")
    print(confusion_matrix(yte, pred))

if __name__ == "__main__":
    main()
