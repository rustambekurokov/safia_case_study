"""Pre-specified candidate models (M1-M6). Transforms: log1p for counts/pops AND distances
(distances use log1p, not log, because dist_bazaar_m / dist_mall_m contain exact zeros); flags as is."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from safia_model.data import NEW4
from safia_model.evaluate import CoordKNN

SET_A = [
    "dist_nearest_safia_m",
    "shops_300m",
    "food_places_300m",
    "bakeries_1km",
    "pop_1km",
    "pop_5km",
    "kindergartens_1km",
    "offices_1km",
    "bus_stops_300m",
    "metro_within_800m",
    "in_mall",
    "is_tashkent",
]
SET_B = SET_A + ["schools_1km", "universities_1km", "hospitals_1km", "dist_bazaar_m", "dist_mall_m", "dist_main_road_m"]
NEW6 = [
    "dist_major_road_m",
    "major_intersection_300m",
    "major_road_len_500m",
    "dormitories_1km",
    "universities_500m",
    "anchors_300m",
]
SET_A2 = SET_A + NEW6
NEW7 = [
    "parking_300m",
    "dist_parking_m",
    "footpath_len_500m",
    "walk_intersections_500m",
    "crossings_300m",
    "on_main_street",
    "corner_site",
]
SET_A3 = SET_A + NEW7
SET_A4 = SET_A + NEW4
SET_A24 = SET_A2 + NEW4
# the 0/1 flag and the already-logged age enter as is; night lights and built volume get log1p
NOLOG = {"open_24_7", "log_age_months"}
FLAGS = {
    "open_24_7",
    "metro_within_800m",
    "in_mall",
    "is_tashkent",
    "major_intersection_300m",
    "on_main_street",
    "corner_site",
}


class Prep(BaseEstimator, TransformerMixin):
    """Select columns and apply log1p to everything except 0/1 flags."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        A = X[list(self.cols)].to_numpy(float).copy()
        for j, c in enumerate(self.cols):
            if c not in FLAGS and c not in NOLOG:
                A[:, j] = np.log1p(A[:, j])
        return A


ALPHAS = np.logspace(-1, 3, 17)


def _pipeline(cols, estimator):
    return Pipeline([("prep", Prep(cols)), ("sc", StandardScaler()), ("m", estimator)])


def ridge(cols):
    return _pipeline(cols, RidgeCV(alphas=ALPHAS))


def lasso(cols):
    return _pipeline(cols, LassoCV(cv=KFold(5, shuffle=True, random_state=0), random_state=0, max_iter=100000))


def rf(cols):
    return _pipeline(cols, RandomForestRegressor(500, min_samples_leaf=5, max_features=1 / 3, random_state=0, n_jobs=4))


def gbm(cols):
    return _pipeline(
        cols,
        GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=0),
    )


class GroupRidge(BaseEstimator, RegressorMixin):
    """Ridge on `cols` whose RidgeCV alpha is chosen by GroupKFold(5) inside the training fold. Groups are rebuilt from the
    training rows' lat/lon (single linkage at 300 m, as in make_splits.py), so X must carry lat and lon columns."""

    def __init__(self, cols=None, thresh_m=300.0):
        self.cols, self.thresh_m = cols, thresh_m

    def _groups(self, X):
        R = 6371008.8
        la, lo = np.radians(X["lat"].to_numpy(float)), np.radians(X["lon"].to_numpy(float))
        a = (
            np.sin((la[:, None] - la[None]) / 2) ** 2
            + np.cos(la)[:, None] * np.cos(la)[None] * np.sin((lo[:, None] - lo[None]) / 2) ** 2
        )
        return connected_components(csr_matrix(2 * R * np.arcsin(np.sqrt(a)) <= self.thresh_m))[1]

    def fit(self, X, y):
        groups = self._groups(X)
        cv = list(GroupKFold(min(5, len(set(groups)))).split(X, y, groups))
        self.pipe_ = _pipeline(self.cols, RidgeCV(alphas=ALPHAS, cv=cv)).fit(X, y)
        return self

    def predict(self, X):
        return self.pipe_.predict(X)

    @property
    def named_steps(self):
        return self.pipe_.named_steps


class TwoPart(BaseEstimator, RegressorMixin):
    """log sales = ridge(log orders) + ridge(log check) + constant (fitted on the training fold).
    fit(X, y, aux=DataFrame[log orders_avg, log check_index]) -- aux only for training rows."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y, aux):
        self.o_ = ridge(self.cols).fit(X, aux.iloc[:, 0].to_numpy())
        self.c_ = ridge(self.cols).fit(X, aux.iloc[:, 1].to_numpy())
        self.k_ = np.mean(np.asarray(y) - self.o_.predict(X) - self.c_.predict(X))
        return self

    def parts(self, X):
        return self.o_.predict(X), self.c_.predict(X)

    def predict(self, X):
        o, c = self.parts(X)
        return o + c + self.k_


class Blend(BaseEstimator, RegressorMixin):
    """Average in log space of ridge(set A) and coordinate kNN(k=10)."""

    def __init__(self, cols=None, k=10, group_aware=False):
        self.cols, self.k, self.group_aware = cols, k, group_aware

    def fit(self, X, y):
        self.a_ = (GroupRidge(self.cols) if self.group_aware else ridge(self.cols)).fit(X, y)
        self.b_ = CoordKNN(self.k).fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.a_.predict(X) + 0.5 * self.b_.predict(X)


LATLON = ["lat", "lon"]
REGISTRY = {
    "m1": dict(desc="M1 Ridge on set A", build=lambda: ridge(SET_A), feature_cols=SET_A, input_cols=SET_A),
    "m4": dict(desc="M4 Random forest on set B", build=lambda: rf(SET_B), feature_cols=SET_B, input_cols=SET_B),
    "m6": dict(
        desc="M6 Blend of M1 and coordinates kNN k=10",
        build=lambda: Blend(SET_A, 10),
        feature_cols=SET_A,
        input_cols=SET_A + LATLON,
    ),
    "m1b": dict(
        desc="M1b Ridge (group-aware alpha) on set A2 (A + 6 round-2 features)",
        build=lambda: GroupRidge(SET_A2),
        feature_cols=SET_A2,
        input_cols=SET_A2 + LATLON,
    ),
    "m4b": dict(desc="M4b Random forest on set A2", build=lambda: rf(SET_A2), feature_cols=SET_A2, input_cols=SET_A2),
    "m6b": dict(
        desc="M6b Blend of M1b and coordinates kNN k=10",
        build=lambda: Blend(SET_A2, 10, True),
        feature_cols=SET_A2,
        input_cols=SET_A2 + LATLON,
    ),
    "m1c": dict(
        desc="M1c Ridge on set A3 (A + 7 round-3 location features)",
        build=lambda: ridge(SET_A3),
        feature_cols=SET_A3,
        input_cols=SET_A3,
    ),
    "m4c": dict(desc="M4c Random forest on set A3", build=lambda: rf(SET_A3), feature_cols=SET_A3, input_cols=SET_A3),
    "m6c": dict(
        desc="M6c Blend of M1c and coordinates kNN k=10",
        build=lambda: Blend(SET_A3, 10),
        feature_cols=SET_A3,
        input_cols=SET_A3 + LATLON,
    ),
    "m1d": dict(
        desc="M1d Ridge on set A4 (A + open_24_7, log_age_months, night_lights_500m, built_volume_500m)",
        build=lambda: ridge(SET_A4),
        feature_cols=SET_A4,
        input_cols=SET_A4,
    ),
    "m4d": dict(
        desc="M4d Random forest on set A2 + the 4 round-4 features",
        build=lambda: rf(SET_A24),
        feature_cols=SET_A24,
        input_cols=SET_A24,
    ),
    "m6d": dict(
        desc="M6d Blend of M1d and coordinates kNN k=10",
        build=lambda: Blend(SET_A4, 10),
        feature_cols=SET_A4,
        input_cols=SET_A4 + LATLON,
    ),
}
