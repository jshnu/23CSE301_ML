import pandas as pd

def label_encode(column):
    column = column.astype(str).fillna("Missing")
    values = column.unique()
    mapping = {value: index for index, value in enumerate(values)}
    return column.map(mapping)

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

label_df = df.copy()

categorical = label_df.select_dtypes(include=["object", "string"]).columns

for col in categorical:
    label_df[col] = label_encode(label_df[col])

onehot_df = pd.get_dummies(df, columns=categorical, dtype=int)

print("Original dimensionality:", df.shape)
print("After Label Encoding:", label_df.shape)
print("After One-Hot Encoding:", onehot_df.shape)
