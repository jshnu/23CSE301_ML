import pandas as pd
import numpy as np

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def calculate_entropy(labels):

    _, counts = np.unique(
        labels,
        return_counts=True
    )

    probabilities = counts / len(labels)

    return -np.sum(
        probabilities *
        np.log2(probabilities)
    )


def calculate_information_gain(
    labels,
    left_labels,
    right_labels
):

    parent_entropy = calculate_entropy(
        labels
    )

    left_weight = (
        len(left_labels) /
        len(labels)
    )

    right_weight = (
        len(right_labels) /
        len(labels)
    )

    child_entropy = (
        left_weight *
        calculate_entropy(left_labels)
        +
        right_weight *
        calculate_entropy(right_labels)
    )

    return parent_entropy - child_entropy


def find_best_split(X, y):

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):

        values = np.unique(
            X[:, feature]
        )

        if len(values) > 20:

            thresholds = np.percentile(
                values,
                np.linspace(5, 95, 19)
            )

        else:

            thresholds = (
                values[:-1] +
                values[1:]
            ) / 2

        for threshold in thresholds:

            left_mask = (
                X[:, feature] <= threshold
            )

            right_mask = ~left_mask

            if (
                np.sum(left_mask) == 0 or
                np.sum(right_mask) == 0
            ):
                continue

            left_labels = y[left_mask]
            right_labels = y[right_mask]

            gain = calculate_information_gain(
                y,
                left_labels,
                right_labels
            )

            if gain > best_gain:

                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return (
        best_feature,
        best_threshold,
        best_gain
    )


def majority_class(labels):

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    return classes[
        np.argmax(counts)
    ]


def build_tree(
    X,
    y,
    depth=0,
    max_depth=4,
    min_samples_split=10
):

    if len(np.unique(y)) == 1:

        return {
            "type": "leaf",
            "class": y[0]
        }

    if (
        depth >= max_depth or
        len(y) < min_samples_split
    ):

        return {
            "type": "leaf",
            "class": majority_class(y)
        }

    (
        feature,
        threshold,
        gain
    ) = find_best_split(X, y)

    if (
        feature is None or
        gain <= 0
    ):

        return {
            "type": "leaf",
            "class": majority_class(y)
        }

    left_mask = (
        X[:, feature] <= threshold
    )

    right_mask = ~left_mask

    left_tree = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth,
        min_samples_split
    )

    right_tree = build_tree(
        X[right_mask],
        y[right_mask],
        depth + 1,
        max_depth,
        min_samples_split
    )

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
        "gain": gain,
        "left": left_tree,
        "right": right_tree
    }


def predict_sample(sample, tree):

    if tree["type"] == "leaf":
        return tree["class"]

    if (
        sample[tree["feature"]]
        <= tree["threshold"]
    ):

        return predict_sample(
            sample,
            tree["left"]
        )

    return predict_sample(
        sample,
        tree["right"]
    )


def predict_dataset(X, tree):

    predictions = []

    for sample in X:

        predictions.append(
            predict_sample(
                sample,
                tree
            )
        )

    return np.array(predictions)


data = pd.read_csv(DATASET)

feature_columns = [
    column
    for column in data.columns
    if column.startswith("CQCC_")
]

X = data[
    feature_columns
].to_numpy()

y = data[
    "Label"
].astype(str).to_numpy()


tree = build_tree(
    X,
    y,
    max_depth=4,
    min_samples_split=10
)


predictions = predict_dataset(
    X,
    tree
)


accuracy = np.mean(
    predictions == y
)


print("Samples:", len(y))
print("Features:", X.shape[1])
print("Training accuracy:", accuracy)
print("\nRoot feature:")
print(
    feature_columns[
        tree["feature"]
    ]
)

print(
    "Root threshold:",
    tree["threshold"]
)

print(
    "Root information gain:",
    tree["gain"]
)