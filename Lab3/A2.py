import pandas as pd

def label_encode(column):
    column = column.astype(str)
    values = column.unique()
    mapping = {}

    for i, value in enumerate(values):
        mapping[value] = i

    return column.map(mapping), mapping

def one_hot_encode(column):
    column = column.astype(str)
    values = column.unique()
    data = {}

    for value in values:
        data[value] = (column == value).astype(int)

    return pd.DataFrame(data)

df = pd.read_excel("data.xlsx", sheet_name="marketing_campaign")

categorical = df.select_dtypes(include=["object", "string", "datetime"]).columns

for col in categorical:
    encoded, mapping = label_encode(df[col])
    print(col)
    print(mapping)
    print(encoded.head())

for col in categorical:
    print(one_hot_encode(df[col]).head())
