import pandas as pd, numpy as np

PATH = "train_ver2.csv"
ID, N_CUST, SEED = "ncodpers", 50_000, 42

# Pass 1 — collect customer IDs only
ids = pd.concat([c[ID] for c in pd.read_csv(PATH, usecols=[ID], chunksize=1_000_000)])
ids = ids.unique()
keep = set(pd.Series(ids).sample(N_CUST, random_state=SEED))
print(f"{len(ids):,} customers total → sampling {len(keep):,}")

# Pass 2 — keep all rows for sampled customers
parts = [c[c[ID].isin(keep)] for c in pd.read_csv(PATH, chunksize=500_000, low_memory=False)]
df = pd.concat(parts, ignore_index=True)

# Coerce the messy numerics
for col in ["age", "antiguedad", "renta"]:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan   # -999999 sentinel

df.to_parquet("santander_sample.parquet", index=False)
print(df.shape, df["fecha_dato"].nunique(), "months")