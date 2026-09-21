import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def entropy(y):

    _, counts = np.unique(
        y,
        return_counts=True
    )

    p = counts / len(y)

    return -np.sum(
        p * np.log2(p)
    )


def information_gain(y, left, right):

    parent = entropy(y)

    weighted = (
        len(left) / len(y) * entropy(left)
        +
        len(right) / len(y) * entropy(right)
    )

    return parent - weighted


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
                left_mask.sum() == 0 or
                right_mask.sum() == 0
            ):
                continue

            gain = information_gain(
                y,
                y[left_mask],
                y[right_mask]
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


def majority_class(y):

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    return classes[
        np.argmax(counts)
    ]


def build_tree(
    X,
    y,
    depth=0,
    max_depth=3
):

    if len(np.unique(y)) == 1:

        return {
            "type": "leaf",
            "class": y[0]
        }

    if depth >= max_depth:

        return {
            "type": "leaf",
            "class": majority_class(y)
        }

    feature, threshold, gain = find_best_split(
        X,
        y
    )

    if feature is None or gain <= 0:

        return {
            "type": "leaf",
            "class": majority_class(y)
        }

    left_mask = (
        X[:, feature] <= threshold
    )

    right_mask = ~left_mask

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            max_depth
        ),
        "right": build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            max_depth
        )
    }


def draw_tree(
    tree,
    feature_names,
    x=0.5,
    y=1.0,
    dx=0.25,
    level=0
):

    if tree["type"] == "leaf":

        plt.text(
            x,
            y,
            "Class = " + str(tree["class"]),
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round"
            )
        )

        return

    feature_name = feature_names[
        tree["feature"]
    ]

    text = (
        feature_name +
        "\n<= " +
        f"{tree['threshold']:.3f}"
    )

    plt.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round"
        )
    )

    child_y = y - 0.15

    plt.plot(
        [x, x - dx],
        [y - 0.03, child_y + 0.03]
    )

    plt.plot(
        [x, x + dx],
        [y - 0.03, child_y + 0.03]
    )

    draw_tree(
        tree["left"],
        feature_names,
        x - dx,
        child_y,
        dx / 2,
        level + 1
    )

    draw_tree(
        tree["right"],
        feature_names,
        x + dx,
        child_y,
        dx / 2,
        level + 1
    )


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
    max_depth=3
)


plt.figure(
    figsize=(18, 10)
)

draw_tree(
    tree,
    feature_columns
)

plt.axis("off")

plt.title(
    "ASVspoof 2017 V2 Decision Tree"
)

plt.savefig(
    "A6_Decision_Tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()