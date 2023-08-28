import pandas as pd
from pathlib import Path

## Load ds000030 participants.csv and keep only required subjects and columns

root_p = Path("__file__").resolve().parents[1] / "data" / "ds000030"

df = pd.read_csv(root_p / "participants.csv")

# Keep only rows where diagnosis is CONTROL or SCHZ
df = df[df["diagnosis"].isin(["CONTROL", "SCHZ"])]

# Add columns SZ and CON
df["SZ"] = (df["diagnosis"] == "SCHZ").astype(int)
df["CON"] = (df["diagnosis"] == "CONTROL").astype(int)

# Add GROUP column
df["GROUP"] = df["diagnosis"]
df["GROUP"] = df["GROUP"].replace({"SCHZ": "SZ", "CONTROL": "CON"})

# Rename some columns and map values for sex
df.rename(columns={"gender": "SEX", "age": "AGE"}, inplace=True)
df["SEX"] = df["SEX"].map({"M": 0, "F": 1})

# Add COHORT and SITE columns
df["COHORT"] = "ds000030"
df["SITE"] = 0

cols = ["participant_id", "AGE", "SEX", "SITE", "SZ", "CON", "COHORT", "GROUP"]
df = df[cols]

# Save results
df.to_csv(root_p / "ds000030_participants.csv", index=False)
