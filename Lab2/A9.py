import pandas as pd
import numpy as np


def loadData():
    return pd.read_excel("data.xlsx", sheet_name="thyroid0387_UCI")


def findOutliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[
        (df[column] < lower) |
        (df[column] > upper)
    ][column].count()


def imputeData(df):

    newDf = df.copy()

    for col in newDf.columns:

        if newDf[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(newDf[col]):

            outliers = findOutliers(newDf, col)

            if outliers == 0:
                newDf[col] = newDf[col].fillna(
                    newDf[col].mean()
                )
            else:
                newDf[col] = newDf[col].fillna(
                    newDf[col].median()
                )

        else:

            newDf[col] = newDf[col].fillna(
                newDf[col].mode()[0]
            )

    return newDf


def minMaxNormalize(df):

    normalizedDf = df.copy()

    numericColumns = normalizedDf.select_dtypes(
        include=np.number
    ).columns

    for col in numericColumns:

        minimum = normalizedDf[col].min()
        maximum = normalizedDf[col].max()

        if maximum != minimum:

            normalizedDf[col] = (
                (normalizedDf[col] - minimum) /
                (maximum - minimum)
            )

        else:
            normalizedDf[col] = 0

    return normalizedDf


# A9

df = loadData()

# First perform A8 imputation
imputedDf = imputeData(df)

# Then perform A9 normalization
normalizedDf = minMaxNormalize(imputedDf)

print("Normalized data:")
print(normalizedDf)

print()

print("Minimum values after normalization:")
print(
    normalizedDf.select_dtypes(include=np.number).min()
)

print()

print("Maximum values after normalization:")
print(
    normalizedDf.select_dtypes(include=np.number).max()
)
