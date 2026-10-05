"""Evaluation harness. Pass a scikit-learn estimator/Pipeline (or a zero-arg factory
returning one). It is cloned and fitted on log(sales_avg) inside each training fold,
so scaling / imputation / selection / tuning placed in the Pipeline are fold-local.
Predictions are exp()'d and clipped to [1, 100] before scoring.
Only the 104 training branches from data.load_train() are ever used."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold

LO, HI = 1.0, 100.0


def _new(model):
    return clone(model) if isinstance(model, BaseEstimator) else model()


def _metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), LO, HI)
    rho = spearmanr(y, p)[0] if np.std(p) > 0 else np.nan  # undefined for constant predictions
    ly, lp = np.log(y), np.log(p)
    return dict(
        MAE=np.mean(np.abs(y - p)),
        RMSE=np.sqrt(np.mean((y - p) ** 2)),
        Spearman=rho,
        MAE_log=np.mean(np.abs(ly - lp)),
        R2_log=1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2),
    )


def _oof(model, X, ylog, splits, aux=None, collect=None):
    """aux: DataFrame of auxiliary training targets, passed as fit(..., aux=aux_train) for TRAINING rows only.
    collect(fitted, X_test, test_idx) -> info, stored per fold."""
    pred = np.full(len(X), np.nan)
    info = []
    for tr, te in splits:
        m = _new(model)
        m = m.fit(X.iloc[tr], ylog[tr]) if aux is None else m.fit(X.iloc[tr], ylog[tr], aux=aux.iloc[tr])
        pred[te] = np.exp(m.predict(X.iloc[te]))
        if collect is not None:
            info.append(collect(m, X.iloc[te], te))
    assert not np.isnan(pred).any(), "splits do not cover every row"
    return np.clip(pred, LO, HI), info


def spatial_splits(df):
    b = df["spatial_block"].to_numpy()
    return [(np.where(b != k)[0], np.where(b == k)[0]) for k in sorted(set(b))]


def evaluate(model, df, features, n_repeats=20, seed=0, n_boot=2000, n_splits=5, aux_cols=None, collect=None):
    """df: output of data.load_train(). features: columns passed to the model as X."""
    X, y = df[list(features)], df["sales_avg"].to_numpy(float)
    ylog = np.log(y)
    aux = None if aux_cols is None else np.log(df[list(aux_cols)])
    out = {}
    p_sp, info_sp = _oof(model, X, ylog, spatial_splits(df), aux, collect)
    m_sp = _metrics(y, p_sp)
    rng = np.random.default_rng(seed)
    bm, bs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        bm.append(np.mean(np.abs(y[i] - p_sp[i])))
        bs.append(spearmanr(y[i], p_sp[i])[0] if np.std(p_sp[i]) > 0 and len(set(y[i])) > 1 else np.nan)
    out["spatial"] = dict(
        metrics=m_sp,
        oof=p_sp,
        fold_info=info_sp,
        MAE_ci=tuple(np.percentile(bm, [2.5, 97.5])),
        Spearman_ci=tuple(np.nanpercentile(bs, [2.5, 97.5])) if not np.isnan(m_sp["Spearman"]) else (np.nan, np.nan),
    )
    # repeated grouped CV
    rows, oofs, infos = [], [], []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        p, inf = _oof(model, X, ylog, list(gkf.split(X, ylog, df["group_id"])), aux, collect)
        rows.append(_metrics(y, p))
        oofs.append(p)
        infos += inf
    R = pd.DataFrame(rows)
    out["grouped"] = dict(
        mean=R.mean().to_dict(), std=R.std(ddof=1).to_dict(), per_repeat=R, oof=np.mean(oofs, 0), fold_info=infos
    )
    return out


# ---- baseline estimators (predict in log space) ----
class Centre(BaseEstimator, RegressorMixin):
    """kind: 'mean' = log(mean sales); 'geo' = exp(mean log sales); 'median'.
    by: optional column name; separate centre per value (falls back to global)."""

    def __init__(self, kind="mean", by=None):
        self.kind, self.by = kind, by

    def _c(self, yl):
        return {
            "mean": lambda: np.log(np.mean(np.exp(yl))),
            "geo": lambda: np.mean(yl),
            "median": lambda: np.median(yl),
        }[self.kind]()

    def fit(self, X, y):
        y = np.asarray(y)
        self.g_ = self._c(y)
        self.c_ = {}
        if self.by is not None:
            v = X[self.by].to_numpy()
            self.c_ = {k: self._c(y[v == k]) for k in np.unique(v)}
        return self

    def predict(self, X):
        if self.by is None:
            return np.full(len(X), self.g_)
        return np.array([self.c_.get(k, self.g_) for k in X[self.by].to_numpy()])


class CoordKNN(BaseEstimator, RegressorMixin):
    """Mean log sales of k nearest training-fold branches; distance in metres
    (equirectangular about the fold's mean latitude). Input columns: lat, lon."""

    def __init__(self, k=5):
        self.k = k

    def _xy(self, X):
        R = 6371008.8
        return np.c_[np.radians(X["lon"].to_numpy()) * R * np.cos(self.lat0_), np.radians(X["lat"].to_numpy()) * R]

    def fit(self, X, y):
        self.lat0_ = np.radians(X["lat"].mean())
        self.xy_, self.y_ = self._xy(X), np.asarray(y)
        return self

    def predict(self, X):
        d = np.linalg.norm(self._xy(X)[:, None, :] - self.xy_[None], axis=2)
        idx = np.argsort(d, axis=1)[:, : self.k]
        return self.y_[idx].mean(1)
