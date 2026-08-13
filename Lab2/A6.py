import pandas as pd
import numpy as np


def loadData():
    df = pd.read_excel("data.xlsx", sheet_name="thyroid0387_UCI")
    return df

def encodeData(df):
    encodedDf = df.copy()

    for col in encodedDf.columns:

        if not pd.api.types.is_numeric_dtype(encodedDf[col]):

            values = encodedDf[col].dropna().unique()

            mapping = {}

            for i in range(len(values)):
                mapping[values[i]] = i

            encodedDf[col] = encodedDf[col].map(mapping)

    return encodedDf


def calculateCosine(vector1, vector2):

    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)

    numerator = np.dot(vector1, vector2)

    denominator = (
        np.sqrt(np.dot(vector1, vector1)) *
        np.sqrt(np.dot(vector2, vector2))
    )

    if denominator == 0:
        return 0

    return numerator / denominator



df = loadData()


encodedDf = encodeData(df)

vector1 = encodedDf.iloc[0].fillna(0).values
vector2 = encodedDf.iloc[1].fillna(0).values

cosine = calculateCosine(vector1, vector2)

print("First complete vector:")
print(vector1)

print()

print("Second complete vector:")
print(vector2)

print()

print("Cosine Similarity:", cosine)
