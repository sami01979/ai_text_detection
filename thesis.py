import glob, re
import pandas as pd
from sklearn.model_selection import train_test_split

path =r"D:\ai_thesis\archive\ai_human_master_dataset.csv"
print(path)

parts = []
for chunk in pd.read_csv(path, usecols=["text", "label"], chunksize=100_000):
    parts.append(chunk.sample(frac=0.03, random_state=42))
df = pd.concat(parts, ignore_index=True)
print("sampled:", df.label.value_counts().to_dict())

def clean(t):
    t = re.sub(r"https?://\S+", "", str(t))
    return re.sub(r"\s+", " ", t).strip()

df["text"] = df["text"].apply(clean)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df = df[df["text"].str.split().str.len().between(30, 400)]
df = df.drop_duplicates(subset="text").reset_index(drop=True)

n = df.label.value_counts().min()
df = pd.concat([df[df.label == c].sample(n, random_state=42) for c in (0, 1)])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("balanced:", df.label.value_counts().to_dict())

print(df.groupby("label").text.apply(lambda s: s.str.split().str.len().mean()))

train, temp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp.label, random_state=42)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print(len(train), len(val), len(test))