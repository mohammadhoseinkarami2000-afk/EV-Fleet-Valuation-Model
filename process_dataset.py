import pandas as pd

print("Loading dataset into memory...")

# 1. Read the full dataset
df = pd.read_csv("cars_dataset.csv")

print(f"Dataset successfully loaded with {len(df):,} total rows.")

# 2. Create an EV-only subset file
if "ev_flag" in df.columns:
    ev_df = df[df["ev_flag"] == 1]
    ev_df.to_csv("ev_cars_subset.csv", index=False)
    print(f"Saved 'ev_cars_subset.csv' with {len(ev_df):,} rows.")

# 3. Create a small random 2,000-row sample file
sample_df = df.sample(n=2000, random_state=42)
sample_df.to_csv("cars_sample_small.csv", index=False)
print(f"Saved 'cars_sample_small.csv' with {len(sample_df):,} rows.")