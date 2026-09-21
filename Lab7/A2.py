import pandas as pd
import numpy as np

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


data = pd.read_csv(DATASET)

labels = data["Label"].astype(str).to_numpy()

entropy = calculate_entropy(labels)

print("Number of samples:", len(data))
print("Genuine samples:", np.sum(labels == "genuine"))
print("Spoof samples:", np.sum(labels == "spoof"))
print("Entropy:", entropy)
