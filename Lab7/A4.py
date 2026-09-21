import pandas as pd
import numpy as np

DATASET = "ASVspoof2017_V2_CQCC_Lab7_500.csv"


def bin_feature(
    values,
    method="equal_width",
    bins=4
):

    values = np.asarray(values)

    if bins < 2:
        raise ValueError(
            "Number of bins must be at least 2."
        )

    if method == "equal_width":

        minimum = np.min(values)
        maximum = np.max(values)

        if minimum == maximum:
            return np.zeros(
                len(values),
                dtype=int
            )

        edges = np.linspace(
            minimum,
            maximum,
            bins + 1
        )

        return np.digitize(
            values,
            edges[1:-1]
        )

    elif method == "frequency":

        ranks = pd.Series(values).rank(
            method="first"
        )

        return pd.qcut(
            ranks,
            q=bins,
            labels=False
        ).to_numpy()

    else:

        raise ValueError(
            "Method must be "
            "'equal_width' or 'frequency'."
        )


data = pd.read_csv(DATASET)

values = data[
    "CQCC_001_Mean"
].to_numpy()


equal_width = bin_feature(
    values,
    "equal_width",
    4
)

frequency = bin_feature(
    values,
    "frequency",
    4
)

default = bin_feature(values)


print("Original values:")
print(values[:10])

print("\nEqual-width bins:")
print(equal_width[:10])

print("\nFrequency bins:")
print(frequency[:10])

print("\nDefault bins:")
print(default[:10])

print("\nEqual-width bin counts:")
print(
    pd.Series(equal_width)
    .value_counts()
    .sort_index()
)

print("\nFrequency bin counts:")
print(
    pd.Series(frequency)
    .value_counts()
    .sort_index()
)
