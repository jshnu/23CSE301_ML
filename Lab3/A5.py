import pandas as pd
import matplotlib.pyplot as plt

def minkowski_distance(a, b, p):
    distance = 0
    for x, y in zip(a, b):
        distance += abs(x - y) ** p
    return distance ** (1 / p)

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical, dtype=int)

numeric = df.select_dtypes(include="number")
numeric = numeric.fillna(numeric.mean())

v1 = numeric.iloc[0].tolist()
v2 = numeric.iloc[1].tolist()

p_values = list(range(1, 11))
distances = []

for p in p_values:
    distances.append(minkowski_distance(v1, v2, p))

print(distances)

plt.plot(p_values, distances, marker="o")
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski Distance")
plt.grid(True)
plt.show()
