import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import librosa
    has_librosa = True
except ImportError:
    has_librosa = False
    print("librosa not installed. Install with: pip install librosa")

base = "."

X_TRAIN_CSV = "X_train.csv"
YTR_CSV = "y_train.csv"
XTE_CSV  = "X_test.csv"
YTE_CSV  = "y_test.csv"

use_csv = True

k = 3

def load_splits_from_csv():
    Xtr = pd.read_csv(X_TRAIN_CSV)
    ytr = pd.read_csv(YTR_CSV).squeeze()
    Xte  = pd.read_csv(XTE_CSV)
    yte  = pd.read_csv(YTE_CSV).squeeze()
    return Xtr, ytr, Xte, yte

def build_splits_from_audio():
    from sklearn.model_selection import train_test_split

    train_dir = os.path.join(base, "ASVspoof2017_V2_train")
    proto   = os.path.join(base, "protocol_V2", "ASVspoof2017_V2_train.trn.txt")

    files, labels = [], []
    with open(proto, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                files.append(parts[0])
                labels.append(parts[1])

    feats = []
    valid_y = []
    for i, (fname, label) in enumerate(zip(files, labels)):
        a_path = os.path.join(train_dir, fname)
        if not os.path.exists(a_path):
            continue
        signal, sr = librosa.load(a_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        feats.append(feat)
        valid_y.append(label)
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(files)} audio files...")

    X = pd.DataFrame(np.array(feats))
    y = pd.Series(valid_y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return Xtr, ytr, Xte, yte

def get_splits():
    if use_csv and os.path.exists(X_TRAIN_CSV):
        print("Loading splits from CSV files...")
        return load_splits_from_csv()
    else:
        print("CSV splits not found. Extracting feats from audio...")
        return build_splits_from_audio()

def main():

    Xtr, ytr, Xte, yte = get_splits()

    print(f"Training set size: {Xtr.shape[0]} samples")
    print(f"Test set size    : {Xte.shape[0]} samples")
    print(f"Class distribution in train:\n{ytr.value_counts()}\n")

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtr, ytr)

    print(f"kNN model trained with k = {k}")

    pred = neigh.predict(Xte)

    accuracy = accuracy_score(yte, pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(yte, pred))

    print("Confusion Matrix:")
    print(confusion_matrix(yte, pred))

if __name__ == "__main__":
    main()
