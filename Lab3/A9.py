import pandas as pd
import numpy as np

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

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical, dtype=int)

numeric = df.select_dtypes(include="number")
numeric = numeric.fillna(numeric.mean())

my_mean = []
my_std = []

for col in numeric.columns:
    values = numeric[col].tolist()
    my_mean.append(mean(values))
    my_std.append(standard_deviation(values))

numpy_mean = numeric.mean().tolist()
numpy_std = numeric.std(ddof=0).tolist()

print("My Mean")
print(my_mean)

print("\nNumPy Mean")
print(numpy_mean)

print("\nMy Standard Deviation")
print(my_std)

print("\nNumPy Standard Deviation")
print(numpy_std)
