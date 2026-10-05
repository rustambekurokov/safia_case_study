"""Predict the sales index of a location (or a CSV of locations) from its surroundings."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safia_model.predict_lib as P  # noqa: E402

TRUE_VALUES = ("1", "1.0", "true", "yes", "y")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--csv")
    ap.add_argument("--out")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default="m6d", help="model key; only m6d ships")
    ap.add_argument(
        "--store-code",
        help="force existing-store treatment as this Safia code (excluded from the network features; "
        "hours/age from store_attributes.csv). Default: automatic if a store is within 30 m",
    )
    ap.add_argument(
        "--new-site",
        action="store_true",
        help="force new-site treatment even next to an existing store "
        "(it then counts as a neighbour; default hours/age)",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--hours",
        choices=["24/7", "day"],
        help="store decision: open around the clock or daytime hours (only model m6d ships). "
        "Default: existing store (see --store-code) -> its listed hours; "
        "else the most common value among training branches",
    )
    ap.add_argument(
        "--age-months",
        type=float,
        help="store history: months since the branch was first listed (only model m6d ships). "
        "Default: existing store -> its value; else the training median",
    )
    a = ap.parse_args()
    if a.age_months is not None and a.age_months < 0:
        ap.error("--age-months must be >= 0")
    if a.new_site and a.store_code:
        ap.error("--new-site and --store-code cannot be used together")
    return ap, a


def to_jsonable(obj):
    """Round-trip through json so numpy scalars become plain numbers."""
    default = lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
    return json.loads(json.dumps(obj, default=default, allow_nan=True))


def predict_csv(ap, a, model, meta):
    d = pd.read_csv(a.csv)
    if not {"lat", "lon"} <= set(d.columns):
        ap.error("csv needs lat and lon columns")
    if "exclude_code" in d and "store_code" not in d:  # old column name
        d = d.rename(columns={"exclude_code": "store_code"})
    codes = d["store_code"] if "store_code" in d else [None] * len(d)
    new_sites = (
        [False] * len(d) if "new_site" not in d else [str(x).strip().lower() in TRUE_VALUES for x in d["new_site"]]
    )
    hours = d["hours"] if "hours" in d else [None] * len(d)
    ages = d["age_months"] if "age_months" in d else [None] * len(d)
    try:
        res = [
            P.predict_one(
                model,
                meta,
                float(lat),
                float(lon),
                None if pd.isna(code) or code == "" else str(code),
                a.force,
                None if pd.isna(h) else h,
                None if pd.isna(age) else float(age),
                new_site,
            )
            for lat, lon, code, h, age, new_site in zip(d["lat"], d["lon"], codes, hours, ages, new_sites)
        ]
    except ValueError as e:
        ap.error(f"csv row: {e}")

    if a.json:
        print(json.dumps(to_jsonable(res), ensure_ascii=False, indent=1))
        return
    out = d.copy()
    out["prediction"] = [x["prediction"] for x in res]
    out["interval_lo"] = [x.get("lo") for x in res]
    out["interval_hi"] = [x.get("hi") for x in res]
    out["nearest_training_km"] = [x.get("nearest_training_km") for x in res]
    out["site_treatment"] = [
        (x["site"]["mode"] + (" " + x["site"]["code"] if x["site"]["code"] else "")) if "site" in x else None
        for x in res
    ]
    out["n_warnings"] = [len(x["warnings"]) for x in res]
    out["warnings"] = [" | ".join(x["warnings"]) for x in res]
    if any("store_inputs" in x for x in res):
        out["open_24_7_used"] = [x.get("store_inputs", {}).get("open_24_7") for x in res]
        out["age_months_used"] = [x.get("store_inputs", {}).get("age_months") for x in res]
        out["store_inputs_source"] = [" / ".join(x.get("store_inputs", {}).get("sources", {}).values()) for x in res]
    for c in meta["feature_cols"]:
        out["f_" + c] = [x.get("features", {}).get(c) for x in res]
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    print(out[["lat", "lon", "prediction", "interval_lo", "interval_hi", "n_warnings"]].to_string())


def main():
    ap, a = parse_args()
    model, meta = P.load_bundle(a.model)
    if a.csv:
        predict_csv(ap, a, model, meta)
        return
    if a.lat is None or a.lon is None:
        ap.error("give --lat and --lon, or --csv")
    try:
        r = P.predict_one(model, meta, a.lat, a.lon, a.store_code, a.force, a.hours, a.age_months, a.new_site)
    except ValueError as e:
        ap.error(str(e))
    print(json.dumps(to_jsonable(r), ensure_ascii=False, indent=1) if a.json else P.format_text(r, meta))


if __name__ == "__main__":
    main()
