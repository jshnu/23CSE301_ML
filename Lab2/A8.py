    import pandas as pd
    import numpy as np


    def loadData():
        df = pd.read_excel("data.xlsx", sheet_name="thyroid0387_UCI")
        return df

def findNumericOutliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = df[
        (df[column] < lower) |
        (df[column] > upper)
    ][column].count()

    return count


def imputeData(df):

    newDf = df.copy()

    for col in newDf.columns:

        if newDf[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(newDf[col]):

            outliers = findNumericOutliers(newDf, col)

            if outliers == 0:
                value = newDf[col].mean()
                newDf[col] = newDf[col].fillna(value)

            else:
                value = newDf[col].median()
                newDf[col] = newDf[col].fillna(value)

        else:

            value = newDf[col].mode()[0]
            newDf[col] = newDf[col].fillna(value)

    return newDf


df = loadData()

print("Missing values before imputation:")
print(df.isnull().sum())

print()

imputedDf = imputeData(df)

print("Missing values after imputation:")
print(imputedDf.isnull().sum())
