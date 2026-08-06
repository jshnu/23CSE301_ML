import pandas as pd
import matplotlib.pyplot as plt

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    total = 0
    for value in values:
        total += (value - m) ** 2
    return total / len(values)

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

feature = "Income"

values = pd.to_numeric(df[feature], errors="coerce").dropna().tolist()

print("Mean:", mean(values))
print("Variance:", variance(values))

plt.hist(values, bins=10)
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title("Histogram of " + feature)
plt.grid(True)
plt.show()
