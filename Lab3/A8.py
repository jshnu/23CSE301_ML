import pandas as pd

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    total = 0
    for value in values:
        total += (value - m) ** 2
    return total / len(values)

def standard_deviation(values):
    return variance(values) ** 0.5

def statistics(df):
    means = {}
    variances = {}
    stds = {}

    for column in df.columns:
        values = df[column].tolist()
        means[column] = mean(values)
        variances[column] = variance(values)
        stds[column] = standard_deviation(values)

    return means, variances, stds

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical, dtype=int)

numeric = df.select_dtypes(include="number")
numeric = numeric.fillna(numeric.mean())

means, variances, stds = statistics(numeric)

print("Mean")
print(means)

print("\nVariance")
print(variances)

print("\nStandard Deviation")
print(stds)
