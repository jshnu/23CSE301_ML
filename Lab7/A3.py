import pandas as pd
import numpy as np

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


def equal_width_binning(values, bins=4):
    minimum = np.min(values)
    maximum = np.max(values)

    if minimum == maximum:
        return np.zeros(len(values), dtype=int)

    edges = np.linspace(
        minimum,
        maximum,
        bins + 1
    )

    return np.digitize(
        values,
        edges[1:-1]
    )


def calculate_information_gain(feature, labels, bins=4):

    parent_entropy = calculate_entropy(labels)

    binned_feature = equal_width_binning(
        feature,
        bins
    )

    weighted_entropy = 0

    for value in np.unique(binned_feature):

        subset = labels[
            binned_feature == value
        ]

        weight = len(subset) / len(labels)

        weighted_entropy += (
            weight *
            calculate_entropy(subset)
        )

    return parent_entropy - weighted_entropy


data = pd.read_csv(DATASET)

feature_columns = [
    column
    for column in data.columns
    if column.startswith("CQCC_")
]

labels = data["Label"].astype(str).to_numpy()

results = []

for feature in feature_columns:

    gain = calculate_information_gain(
        data[feature].to_numpy(),
        labels,
        bins=4
    )

    results.append(
        [feature, gain]
    )


results = pd.DataFrame(
    results,
    columns=[
        "Feature",
        "Information_Gain"
    ]
)

results = results.sort_values(
    "Information_Gain",
    ascending=False
)


print("\nTop 10 features:\n")
print(
    results.head(10).to_string(
        index=False
    )
)

print(
    "\nRoot feature:",
    results.iloc[0]["Feature"]
)

print(
    "Information gain:",
    results.iloc[0]["Information_Gain"]
)
