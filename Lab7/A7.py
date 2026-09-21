import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def entropy(y):

    _, counts = np.unique(
        y,
        return_counts=True
    )

    probabilities = counts / len(y)

    return -np.sum(
        probabilities *
        np.log2(probabilities)
    )


def calculate_information_gain(
    feature,
    y,
    bins=4
):

    minimum = np.min(feature)
    maximum = np.max(feature)

    if minimum == maximum:
        return 0

    edges = np.linspace(
        minimum,
        maximum,
        bins + 1
    )

    binned = np.digitize(
        feature,
        edges[1:-1]
    )

    parent_entropy = entropy(y)

    weighted_entropy = 0

    for value in np.unique(binned):

        subset = y[
            binned == value
        ]

        weighted_entropy += (
            len(subset) / len(y)
            * entropy(subset)
        )

    return (
        parent_entropy -
        weighted_entropy
    )


data = pd.read_csv(DATASET)

feature_columns = [
    column
    for column in data.columns
    if column.startswith("CQCC_")
]

y = data[
    "Label"
].map({
    "genuine": 0,
    "spoof": 1
}).to_numpy()


gains = []

for feature in feature_columns:

    gain = calculate_information_gain(
        data[feature].to_numpy(),
        y
    )

    gains.append(gain)


ranking = np.argsort(gains)[::-1]


feature_1 = feature_columns[
    ranking[0]
]

feature_2 = feature_columns[
    ranking[1]
]


print("Feature 1:", feature_1)
print("Feature 2:", feature_2)


X = data[
    [feature_1, feature_2]
].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


print(
    "Test accuracy:",
    model.score(
        X_test,
        y_test
    )
)


x_min = X[:, 0].min()
x_max = X[:, 0].max()

y_min = X[:, 1].min()
y_max = X[:, 1].max()

x_margin = (
    x_max - x_min
) * 0.05

y_margin = (
    y_max - y_min
) * 0.05


xx, yy = np.meshgrid(
    np.linspace(
        x_min - x_margin,
        x_max + x_margin,
        400
    ),
    np.linspace(
        y_min - y_margin,
        y_max + y_margin,
        400
    )
)


grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


prediction = model.predict(
    grid
)


prediction = prediction.reshape(
    xx.shape
)


plt.figure(
    figsize=(10, 7)
)

plt.contourf(
    xx,
    yy,
    prediction,
    alpha=0.25
)


plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Genuine",
    edgecolor="black"
)


plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Spoof",
    edgecolor="black"
)


plt.xlabel(feature_1)
plt.ylabel(feature_2)

plt.title(
    "Decision Boundary Using Two CQCC Features"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "A7_Decision_Boundary.png",
    dpi=300
)

plt.show()