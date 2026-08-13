import pandas as pd
import numpy as np

df = pd.read_excel("data.xlsx", sheet_name="thyroid0387_UCI")

numericColumns = df.select_dtypes(include=np.number).columns

for col in numericColumns:
    mean = df[col].mean()
    variance = df[col].var()

    print(f"{col}: Mean = {mean}, Variance = {variance}")
