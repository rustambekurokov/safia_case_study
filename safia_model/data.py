"""Single loader for modelling/EDA. Returns ONLY eligible, non-holdout branches.

Holdout rows of branches.csv are dropped while streaming the file row by row
(keyed on the code column); their values are never put into a DataFrame.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

NEW4 = ["open_24_7", "log_age_months", "night_lights_500m", "built_volume_500m"]
NUMERIC = ["lat", "lon", "lat_file", "lon_file", "sales_avg", "orders_avg", "check_index"]


def add_attributes(df):
    """Left-merge store attributes (open_24_7, age_months, age_censored) by code and add log_age_months = log1p(age_months).

    Row set and existing columns are unchanged; empty values stay NaN. age_censored is reported only, not a model feature.
    """
    attrs = pd.read_csv(PROC / "store_attributes.csv")[["code", "open_24_7", "age_months", "age_censored"]]
    out = df.merge(attrs, on="code", how="left", validate="1:1")
    out["log_age_months"] = np.log1p(out["age_months"])
    return out


def complete_new4(df):
    """Rows with all four NEW4 features present (the fair-comparison subset)."""
    return df[df[NEW4].notna().all(axis=1)].reset_index(drop=True)


def load_train() -> pd.DataFrame:
    """The 104 eligible, non-holdout branches with features and store attributes."""
    splits = pd.read_csv(PROC / "splits.csv")
    splits = splits[(splits["eligible"] == 1) & (splits["is_holdout"] == 0)]
    keep = set(splits["code"])
    with open(PROC / "branches.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["code"] in keep]
    branches = pd.DataFrame(rows)
    for c in NUMERIC:
        branches[c] = pd.to_numeric(branches[c])
    branches = branches[["code", "name", "format", "city"] + NUMERIC]
    feats = pd.read_csv(PROC / "features.csv")
    feats = feats[feats["code"].isin(keep)].drop(columns=["coord_status"])
    df = branches.merge(feats, on="code", how="inner", validate="1:1")
    df = df.merge(splits[["code", "group_id", "spatial_block"]], on="code", validate="1:1")
    assert len(df) == 104, len(df)
    return add_attributes(df.reset_index(drop=True))
