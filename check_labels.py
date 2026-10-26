
import pandas as pd

data = pd.read_csv("data/WELFake_Dataset.csv")
data.columns = data.columns.str.strip().str.lower()

print("LABEL COUNTS:")
print(data["label"].value_counts().to_string())

print("\nLABEL 0 SAMPLE:")
print(data.loc[data["label"] == 0, "text"].dropna().head(3).to_string(index=False))

print("\nLABEL 1 SAMPLE:")
print(data.loc[data["label"] == 1, "text"].dropna().head(3).to_string(index=False))
