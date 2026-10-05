"""Prediction for a single location: site resolution, store inputs, warnings and text output."""

import contextlib
import os
import sys
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import safia_model.evaluate as _ev
import safia_model.features as F
import safia_model.intervals as _iv
import safia_model.models as M
from safia_model.evaluate import LO, HI
from safia_model.intervals import conformal_q, interval

ROOT = F.ROOT

# The shipped .joblib files were pickled when these modules were top-level (`models.Blend`, `evaluate.CoordKNN`, ...).
# The old names are aliased to the package modules ONLY while joblib.load runs: aliases left in sys.modules would
# shadow other packages for the rest of the process (e.g. a later `import evaluate` = HuggingFace `evaluate`).
_LEGACY_MODULES = {"models": M, "features": F, "evaluate": _ev, "intervals": _iv}


@contextlib.contextmanager
def _legacy_module_aliases():
    added = [name for name in _LEGACY_MODULES if name not in sys.modules]  # never touch an already imported module
    for name in added:
        sys.modules[name] = _LEGACY_MODULES[name]
    try:
        yield
    finally:
        for name in added:
            if sys.modules.get(name) is _LEGACY_MODULES[name]:
                del sys.modules[name]


UZ_BBOX = (37.1, 45.7, 55.9, 73.3)  # lat_min, lat_max, lon_min, lon_max
FAR_KM = 20.0
LEVEL = 0.8
SAME_STORE_M = 30.0  # a point this close to a Safia store is treated as that store
INTERVAL_NOTE = (
    "The interval is wide because the model explains little of the variation between branches "
    "(cross-validated R2 in log space is roughly 0.05-0.13); treat the number as a rough, compressed guide, "
    "not a forecast. High sellers are systematically under-predicted."
)
INTERVAL_NOTE_STORE_INPUTS = (
    "The interval is wide because the model still explains only part of the variation between branches "
    "(cross-validated R2 in log space is roughly 0.3 for this model, on branches with known 24/7 and age). open_24_7 and age are "
    "STORE DECISIONS / history that you set, not properties of the location; the prediction is conditional on them. "
    "High sellers are still under-predicted."
)


@lru_cache(None)
def _networks():
    return F.load_safia(F.DEFAULT_SAFIA), F.load_safia(F.KNOWN_SAFIA)


@lru_cache(None)
def _stores():
    """All known stores, one row per code, with name / open_24_7 / age_months from store_attributes.csv where listed."""
    net = pd.concat(_networks()).dropna(subset=["code"]).drop_duplicates("code")
    attrs = pd.read_csv(os.path.join(ROOT, "data/processed/store_attributes.csv"))[
        ["code", "name", "open_24_7", "age_months"]
    ]
    return net.merge(attrs, on="code", how="left")


def _hav_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(a))


def resolve_site(lat, lon, store_code=None, new_site=False):
    """Decide whether a point is an existing Safia store; returns a dict with mode 'existing' or 'new'.

    store_code forces 'existing' (ValueError if unknown); new_site forces 'new'; otherwise 'existing' iff a store is
    within SAME_STORE_M (the nearest one; the rest are listed in `others`). For a new site `nearest` is the nearest
    Safia store, which counts as a neighbour.
    """
    if new_site and store_code:
        raise ValueError("new_site and store_code are mutually exclusive")
    stores = _stores()
    dist_m = _hav_km(lat, lon, stores["lat"].to_numpy(), stores["lon"].to_numpy()) * 1000

    def label(i):
        name = stores["name"].iloc[i]
        return f"{stores['code'].iloc[i]} {name if pd.notna(name) else ''}".strip()

    if store_code:
        matches = np.flatnonzero(stores["code"].astype(str).to_numpy() == str(store_code))
        if not len(matches):
            raise ValueError(f"unknown store code {store_code!r} (not in safia_network.csv or the known branches)")
        j = int(matches[0])
        how = "forced by --store-code / store_code column"
    elif not new_site and dist_m.min() <= SAME_STORE_M:
        j = int(np.argmin(dist_m))
        how = f"within {SAME_STORE_M:.0f} m"
    else:
        j = int(np.argmin(dist_m))
        note = (
            "forced by --new-site / new_site column; " if new_site else ""
        ) + f"no Safia store within {SAME_STORE_M:.0f} m"
        nearest = dict(code=stores["code"].iloc[j], label=label(j), dist_m=float(dist_m[j]))
        return dict(mode="new", code=None, name=None, dist_m=None, others=[], nearest=nearest, note=note)
    others = [f"{label(i)} ({dist_m[i]:.0f} m)" for i in np.argsort(dist_m) if i != j and dist_m[i] <= SAME_STORE_M]
    return dict(
        mode="existing",
        code=str(stores["code"].iloc[j]),
        name=label(j),
        dist_m=float(dist_m[j]),
        others=others,
        nearest=None,
        note=how,
        row=stores.iloc[j],
    )


def resolve_store_inputs(meta, site, hours=None, age_months=None):
    """Return (values, sources) for open_24_7 and age_months.

    Priority: explicit argument > the existing store's row in store_attributes.csv (only if `site` is an existing
    store) > default (most common open_24_7 / median age among the model's training branches).
    """
    train = meta["train_table"]
    default_open = int(train["open_24_7"].mode().iloc[0])
    default_age = float(np.expm1(train["log_age_months"].median()))
    values, sources = {}, {}
    store = site["row"] if site["mode"] == "existing" else None
    where = f"existing store {site['name']}, data/processed/store_attributes.csv" if store is not None else None

    if hours is not None:
        values["open_24_7"] = 1 if str(hours) == "24/7" else 0
        sources["open_24_7"] = "user (--hours)"
    elif store is not None and pd.notna(store["open_24_7"]):
        values["open_24_7"] = int(store["open_24_7"])
        sources["open_24_7"] = where
    else:
        values["open_24_7"] = default_open
        sources["open_24_7"] = "DEFAULT: most common value among training branches" + (
            f" (store {store['code']} has no hours listed)" if store is not None else " (new site)"
        )

    if age_months is not None:
        values["age_months"] = float(age_months)
        sources["age_months"] = "user (--age-months)"
    elif store is not None and pd.notna(store["age_months"]):
        values["age_months"] = float(store["age_months"])
        sources["age_months"] = where + " (first listed on the website, not the true opening date)"
    else:
        values["age_months"] = default_age
        sources["age_months"] = "DEFAULT: median age_months among training branches" + (
            f" (store {store['code']} has no age)" if store is not None else " (new site)"
        )
    return values, sources


def load_bundle(key="m1"):
    """Load models/final_<key>.joblib and prepare the interval width and the similarity space."""
    with _legacy_module_aliases():
        bundle = joblib.load(os.path.join(ROOT, "models", f"final_{key}.joblib"))
    meta = bundle["meta"]
    meta["q"] = conformal_q(np.abs(meta["oof_log_residuals"]), LEVEL)
    prep = M.Prep(meta["feature_cols"])
    Z = prep.transform(meta["train_table"])
    meta["_prep"], meta["_sc"] = prep, StandardScaler().fit(Z)
    meta["_Z"] = meta["_sc"].transform(Z)
    return bundle["model"], meta


def predict_one(model, meta, lat, lon, store_code=None, force=False, hours=None, age_months=None, new_site=False):
    """Predict for one point; returns a dict with prediction, interval, warnings and supporting information.

    store_code forces 'existing store' treatment (ValueError if unknown), new_site forces 'new site';
    with neither, the 30 m rule of resolve_site applies.
    """
    out = dict(lat=lat, lon=lon, warnings=[], prediction=None)
    if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        out["warnings"].append("invalid coordinates")
        return out
    network, network_known = _networks()
    site = resolve_site(lat, lon, store_code, new_site)
    out["site"] = {k: v for k, v in site.items() if k != "row"}
    if site["mode"] == "existing":
        out["exclude_code"] = site["code"]
        if site["others"]:
            out["warnings"].append(
                f"{len(site['others'])} other Safia store(s) within {SAME_STORE_M:.0f} m (e.g. a mall); "
                f"treated as the nearest, {site['code']}. Others: " + "; ".join(site["others"])
            )
        if site["dist_m"] > SAME_STORE_M:
            out["warnings"].append(f"forced store {site['code']} is {site['dist_m']:.0f} m from this point")

    outside = not (UZ_BBOX[0] <= lat <= UZ_BBOX[1] and UZ_BBOX[2] <= lon <= UZ_BBOX[3])
    feats = {}
    if not outside:
        feats = F.compute_features(
            lat, lon, exclude_code=out.get("exclude_code"), safia=network, safia_known=network_known
        )
        out["features"] = feats
        if not np.isfinite(feats.get("pop_10km", np.nan)) or feats["pop_10km"] <= 0:
            outside = True
    if outside:
        out["warnings"].append(
            "POINT OUTSIDE UZBEKISTAN (outside the country bounding box or no WorldPop population cells): "
            "features are not meaningful"
        )
        if not force:
            out["warnings"].append("prediction refused (use --force to override)")
            return out
        if not feats:
            return out

    feature_cols, input_cols = meta["feature_cols"], meta["input_cols"]
    uses_store = "open_24_7" in feature_cols
    if uses_store and feats:
        values, sources = resolve_store_inputs(meta, site, hours, age_months)
        feats["open_24_7"] = values["open_24_7"]
        feats["log_age_months"] = float(np.log1p(values["age_months"]))
        out["store_inputs"] = dict(open_24_7=values["open_24_7"], age_months=values["age_months"], sources=sources)
    elif (hours is not None or age_months is not None) and not uses_store:
        out["warnings"].append("--hours / --age-months ignored: this model does not use them")
    if not all(np.isfinite(feats.get(c, np.nan)) for c in feature_cols):
        out["warnings"].append("some model features are not finite; prediction refused")
        return out

    row = pd.DataFrame([{**feats, "lat": lat, "lon": lon}])
    pred = float(np.clip(np.exp(model.predict(row[input_cols])[0]), LO, HI))
    lo, hi = interval(pred, meta["q"])
    out.update(
        prediction=pred,
        lo=lo,
        hi=hi,
        q=meta["q"],
        interval_level=LEVEL,
        interval_note=INTERVAL_NOTE_STORE_INPUTS if uses_store else INTERVAL_NOTE,
    )

    train = meta["train_table"]
    extrapolated = [
        f"{c}={feats[c]:.4g} outside training range [{train[c].min():.4g}, {train[c].max():.4g}]"
        for c in feature_cols
        if c not in M.FLAGS and (feats[c] < train[c].min() or feats[c] > train[c].max())
    ]
    if extrapolated:
        out["warnings"].append("extrapolation: " + "; ".join(extrapolated))
    dist_km = _hav_km(lat, lon, train["lat"].to_numpy(), train["lon"].to_numpy())
    out["nearest_training_km"] = float(dist_km.min())
    if dist_km.min() > FAR_KM:
        out["warnings"].append(
            f"farther than {FAR_KM:.0f} km from every training branch (nearest {dist_km.min():.0f} km): "
            "location is outside the area the model was trained on"
        )
    if not feats["is_tashkent"]:
        out["warnings"].append(
            "outside Tashkent: OSM coverage (bus stops, bakeries, hospitals, kindergartens, shops, food) "
            "is likely thinner, so counts may understate reality"
        )

    z = meta["_sc"].transform(meta["_prep"].transform(pd.DataFrame([feats])))[0]
    feature_dist = np.linalg.norm(meta["_Z"] - z, axis=1)
    out["similar"] = [
        dict(
            code=train.code.iloc[i],
            name=train.name.iloc[i],
            city=train.city.iloc[i],
            feature_distance=float(feature_dist[i]),
            distance_km=float(dist_km[i]),
        )
        for i in np.argsort(feature_dist)[:3]
    ]

    steps = getattr(model, "named_steps", None)
    if steps is not None and hasattr(steps.get("m"), "coef_") and "sc" in steps:
        z_row = steps["sc"].transform(steps["prep"].transform(row))[0]
        contributions = steps["m"].coef_ * z_row
        out["contributions"] = {c: float(v) for c, v in zip(feature_cols, contributions)}
    return out


def format_text(r, meta):
    """Human-readable report for one predict_one result."""
    lines = [f"Location: lat {r['lat']:.5f}, lon {r['lon']:.5f}"]
    site = r.get("site")
    if site and site["mode"] == "existing":
        lines.append(
            f"Site: treated as existing store {site['name']} ({site['dist_m']:.0f} m away; {site['note']}); "
            "it is excluded from the Safia network features, as in training"
        )
    elif site:
        n = site["nearest"]
        lines.append(
            f"Site: treated as a new site ({site['note']}); nearest Safia {n['label']} is {n['dist_m']:.0f} m away "
            "and counts as a neighbour"
        )
    if r["prediction"] is None:
        lines.append("Predicted sales index: NOT COMPUTED")
    else:
        lines.append(f"Predicted sales index: {r['prediction']:.1f}  (scale 1-100; model: {meta['description']})")
        lines.append(
            f"{int(r['interval_level'] * 100)}% prediction interval: {r['lo']:.1f} - {r['hi']:.1f}  "
            f"(prediction x/÷ {np.exp(r['q']):.2f}; split-conformal from out-of-fold grouped-CV residuals)"
        )
        lines.append("  " + r["interval_note"])
    if r.get("store_inputs"):
        si = r["store_inputs"]
        lines.append(
            "Store inputs (STORE DECISIONS / history, not location properties; the prediction is conditional on them):"
        )
        lines.append(
            f"  open_24_7 = {si['open_24_7']} ({'24/7' if si['open_24_7'] else 'not 24/7'}) <- {si['sources']['open_24_7']}"
        )
        lines.append(f"  age_months = {si['age_months']:.1f} <- {si['sources']['age_months']}")
    if r.get("features"):
        lines.append("Features:")
        cols = meta["feature_cols"] if r["prediction"] is not None else list(r["features"])
        lines.append("  " + "; ".join(f"{c}={r['features'][c]:.6g}" for c in cols))
    if r.get("contributions"):
        ranked = sorted(r["contributions"].items(), key=lambda kv: -kv[1])
        top_pos = [f"{k} {v:+.3f}" for k, v in ranked if v > 0][:4]
        top_neg = [f"{k} {v:+.3f}" for k, v in ranked[::-1] if v < 0][:4]
        lines.append(
            "Ridge contributions to log sales vs the average training branch (sum of all = %+.3f):"
            % sum(r["contributions"].values())
        )
        lines.append("  top positive: " + (", ".join(top_pos) or "none"))
        lines.append("  top negative: " + (", ".join(top_neg) or "none"))
    if r.get("similar"):
        lines.append("Most similar training branches (standardised feature space; sales not shown):")
        for s in r["similar"]:
            lines.append(
                f"  {s['code']} {s['name']} ({s['city']}): feature distance {s['feature_distance']:.2f}, "
                f"{s['distance_km']:.1f} km away"
            )
    lines.append("Warnings: " + ("none" if not r["warnings"] else ""))
    lines.extend("  - " + w for w in r["warnings"])
    return "\n".join(lines)
