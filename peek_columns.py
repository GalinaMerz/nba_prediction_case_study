import pandas as pd

PATH = "train_ver2.csv"

peek = pd.read_csv(PATH, nrows=5000)
print(peek.shape)
print(list(peek.columns))
print(peek.dtypes.head(25))
peek.head(3)