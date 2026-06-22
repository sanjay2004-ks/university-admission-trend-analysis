"""
dataset_generator.py
Generates 1,000 synthetic graduate admissions records and saves to admissions_data.csv.
"""

import numpy as np
import pandas as pd

SEED = 42
N = 1000

rng = np.random.default_rng(SEED)

# Base scores drawn from realistic distributions
gre = rng.integers(290, 341, size=N)
toefl = rng.integers(92, 121, size=N)
uni_rating = rng.integers(1, 6, size=N)
sop = (rng.integers(2, 11, size=N) / 2).clip(1.0, 5.0)   # 1.0–5.0 in 0.5 steps
lor = (rng.integers(2, 11, size=N) / 2).clip(1.0, 5.0)
cgpa = np.round(rng.uniform(6.0, 10.0, size=N), 2)
research = rng.integers(0, 2, size=N)

# Chance_of_Admit correlated with CGPA, GRE, TOEFL, research
chance = (
    0.02 * (gre - 290) / 50
    + 0.03 * (toefl - 92) / 28
    + 0.35 * (cgpa - 6.0) / 4.0
    + 0.05 * research
    + 0.02 * (uni_rating - 1) / 4
    + rng.normal(0, 0.05, size=N)
)
# Scale to [0.34, 0.97]
chance = 0.34 + 0.63 * (chance - chance.min()) / (chance.max() - chance.min())
chance = np.round(chance.clip(0.34, 0.97), 2)

admitted = (chance >= 0.65).astype(int)

year = rng.integers(2018, 2025, size=N)
streams = rng.choice(["CS", "EE", "ME", "CE", "DS"], size=N)

df = pd.DataFrame({
    "GRE_Score": gre,
    "TOEFL_Score": toefl,
    "University_Rating": uni_rating,
    "SOP": sop,
    "LOR": lor,
    "CGPA": cgpa,
    "Research": research,
    "Chance_of_Admit": chance,
    "Admitted": admitted,
    "Year": year,
    "Stream": streams,
})

out_path = "admissions_data.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved: {out_path}  ({len(df)} rows)")
print(f"Admission rate: {df['Admitted'].mean():.1%}")
print(df.describe().to_string())
