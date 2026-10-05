"""Split-conformal prediction intervals in log space."""

import numpy as np
from sklearn.model_selection import GroupKFold

from safia_model.evaluate import _oof, LO, HI


def oof_log_residuals(build, df, input_cols, n_repeats=20, seed=0, n_splits=5):
    """Signed log(actual) - log(prediction) out-of-fold residuals, one row per grouped-CV repeat."""
    X, ylog = df[list(input_cols)], np.log(df["sales_avg"].to_numpy(float))
    out = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        p, _ = _oof(build(), X, ylog, list(gkf.split(X, ylog, df["group_id"])))
        out.append(ylog - np.log(p))
    return np.array(out)


def conformal_q(abs_res, level=0.8):
    """Split-conformal quantile of absolute residuals."""
    a = np.sort(np.asarray(abs_res).ravel())
    n = len(a)
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(a[k - 1])


def interval(pred, q):
    """Interval [pred / exp(q), pred * exp(q)] clipped to the sales scale."""
    return float(np.clip(pred * np.exp(-q), LO, HI)), float(np.clip(pred * np.exp(q), LO, HI))
