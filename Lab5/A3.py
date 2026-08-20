import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import librosa
    has_librosa = True
except ImportError:
    has_librosa = False
    print("librosa not installed. Install with: pip install librosa")

base = "."

train_dir = os.path.join(base, "ASVspoof2017_V2_train")
proto_file   = os.path.join(base, "protocol_V2", "ASVspoof2017_V2_train.trn.txt")

n_mfcc = 13

max_frames = None

def parse_protocol(path):
    files = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                files.append(parts[0])
                labels.append(parts[1])
    return files, labels

def extract_mfcc_features(a_path, n_mfcc=n_mfcc, max_frames=max_frames, sr=16000):
    if not has_librosa:
        raise ImportError("librosa is required for feature extraction")

    signal, sr = librosa.load(a_path, sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    if max_frames is not None and mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]

    mean_vec = np.mean(mfcc, axis=1)
    std_vec  = np.std(mfcc, axis=1)

    return np.concatenate([mean_vec, std_vec])

def build_dataset(audio_dir, files, labels, feat_csv=None):
    if feat_csv is not None and os.path.exists(feat_csv):
        df = pd.read_csv(feat_csv)
        y = df["label"]
        X = df.drop(columns=["label"])
        return X, y

    feats = []
    valid_y = []
    total = len(files)

    for i, (fname, label) in enumerate(zip(files, labels)):
        a_path = os.path.join(audio_dir, fname)
        if not os.path.exists(a_path):
            print(f"Warning: {a_path} not found, skipping.")
            continue

        feat = extract_mfcc_features(a_path)
        feats.append(feat)
        valid_y.append(label)

    X = pd.DataFrame(np.array(feats))
    y = pd.Series(valid_y)
    return X, y

def main():

    files, labels = parse_protocol(proto_file)
    print(f"Total utterances in train protocol: {len(files)}")
    print(f"Classes found: {set(labels)}")

    classes = np.unique(labels)
    if len(classes) != 2:
        print(f"Warning: Expected 2 classes, found {len(classes)}")
    else:
        print(f"Two classes confirmed: {classes[0]} and {classes[1]}")

    FEATURE_CSV = "asvspoof_features.csv"
    X, y = build_dataset(train_dir, files, labels, feat_csv=FEATURE_CSV)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("\n=== Split Results ===")
    print(f"Training set size: {Xtr.shape[0]} samples")
    print(f"Test set size    : {Xte.shape[0]} samples")
    print(f"Train class distribution:\n{ytr.value_counts()}")
    print(f"Test  class distribution:\n{yte.value_counts()}")

    Xtr.to_csv("X_train.csv", index=False)
    Xte.to_csv("X_test.csv", index=False)
    ytr.to_csv("y_train.csv", index=False, header=["label"])
    yte.to_csv("y_test.csv", index=False, header=["label"])
    print("\nSplits saved to Xtr.csv, Xte.csv, ytr.csv, yte.csv")

if __name__ == "__main__":
    main()
