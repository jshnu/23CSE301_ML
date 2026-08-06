import pandas as pd
import numpy as np

def dot_product(a, b):
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total

def euclidean_norm(a):
    total = 0
    for x in a:
        total += x * x
    return total ** 0.5

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical, dtype=int)

numeric = df.select_dtypes(include="number")
numeric = numeric.fillna(numeric.mean())

A = numeric.iloc[0].to_numpy()
B = numeric.iloc[1].to_numpy()

print("My Dot Product:", dot_product(A, B))
print("NumPy Dot Product:", np.dot(A, B))

print("My Norm:", euclidean_norm(A))
print("NumPy Norm:", np.linalg.norm(A))
