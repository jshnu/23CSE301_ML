import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def loadData():
    df = pd.read_excel("data.xlsx", sheet_name="thyroid0387_UCI")
    return df

def getBinaryColumns(df):
    binaryColumns = []

    for col in df.columns:
        values = df[col].dropna().unique()

        if len(values) == 2:
            binaryColumns.append(col)

    return binaryColumns

def convertBinaryData(df, columns):
    newDf = df[columns].copy()

    for col in columns:

        values = list(newDf[col].dropna().unique())

        if len(values) == 2:
            mapping = {
                values[0]: 0,
                values[1]: 1
            }

            newDf[col] = newDf[col].map(mapping)

    return newDf

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


def calculateJC(vector1, vector2):
    f11 = 0
    f01 = 0
    f10 = 0

    for i in range(len(vector1)):
        if vector1[i] == 1 and vector2[i] == 1:
            f11 += 1
        elif vector1[i] == 0 and vector2[i] == 1:
            f01 += 1
        elif vector1[i] == 1 and vector2[i] == 0:
            f10 += 1

    denominator = f01 + f10 + f11

    if denominator == 0:
        return 0

    return f11 / denominator



def calculateSMC(vector1, vector2):
    f11 = 0
    f00 = 0
    f01 = 0
    f10 = 0

    for i in range(len(vector1)):
        if vector1[i] == 1 and vector2[i] == 1:
            f11 += 1
        elif vector1[i] == 0 and vector2[i] == 0:
            f00 += 1
        elif vector1[i] == 0 and vector2[i] == 1:
            f01 += 1
        elif vector1[i] == 1 and vector2[i] == 0:
            f10 += 1

    return (f11 + f00) / (f00 + f01 + f10 + f11)


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

n

def createSimilarityMatrices(df, n=20):

    binaryColumns = getBinaryColumns(df)
    binaryDf = convertBinaryData(df, binaryColumns)

    encodedDf = encodeData(df)

    JC = np.zeros((n, n))
    SMC = np.zeros((n, n))
    COS = np.zeros((n, n))

    for i in range(n):

        for j in range(n):

            vector1 = binaryDf.iloc[i].fillna(0).values
            vector2 = binaryDf.iloc[j].fillna(0).values

            JC[i][j] = calculateJC(vector1, vector2)
            SMC[i][j] = calculateSMC(vector1, vector2)

            vector1 = encodedDf.iloc[i].fillna(0).values
            vector2 = encodedDf.iloc[j].fillna(0).values

            COS[i][j] = calculateCosine(vector1, vector2)

    return JC, SMC, COS


def plotHeatmap(data, title):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )

    plt.title(title)
    plt.xlabel("Observation")
    plt.ylabel("Observation")
    plt.show()


df = loadData()


JC, SMC, COS = createSimilarityMatrices(df, 20)

print("Jaccard Similarity Matrix")
print(JC)

print()

print("SMC Similarity Matrix")
print(SMC)

print()

print("Cosine Similarity Matrix")
print(COS)

plotHeatmap(JC, "Jaccard Coefficient - First 20 Observations")

plotHeatmap(SMC, "Simple Matching Coefficient - First 20 Observations")

plotHeatmap(COS, "Cosine Similarity - First 20 Observations")
