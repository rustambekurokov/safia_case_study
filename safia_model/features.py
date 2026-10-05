"""Surroundings features for any (lat, lon). Runs offline from data/.

    from safia_model.features import compute_features
    compute_features(41.31, 69.28)                       # brand-new location
    compute_features(41.31, 69.28, exclude_code="W035")  # existing branch: drop itself from Safia counts

All distances are metres computed in a local azimuthal-equidistant projection centred on the query
point (accurate everywhere in Uzbekistan, independent of UTM zone); point counts use haversine.
No sales / orders / check values are used anywhere.
"""

import glob
import os
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
from sklearn.neighbors import BallTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OSM = os.path.join(ROOT, "data/processed/osm/")
POP_TIF = os.path.join(ROOT, "data/raw/uzb_pop_2026_CN_100m_R2025A_v1.tif")
GHSL_DIR = os.path.join(ROOT, "data/raw/ghsl/")  # GHS-BUILT-V R2023A, epoch 2025, 100 m, Mollweide tiles
NL_TIF = os.path.join(
    ROOT, "data/raw/nightlights/nl2024_uzbekistan_x10.tif"
)  # VNL 2024 v2.x via Zenodo 17294744, int16 x10
NL_SCALE = 10.0
DEFAULT_SAFIA = os.path.join(
    ROOT, "data/processed/safia_network.csv"
)  # cleaned full network (see build_safia_network.py)
KNOWN_SAFIA = os.path.join(ROOT, "data/processed/branches.csv")  # the 127 given branches only
R_EARTH = 6371008.8
DEDUP_M = {
    "bus_stops": 25,
    "schools": 30,
    "kindergartens": 30,
    "universities": 30,
    "hospitals": 30,
    "offices": 30,
    "dormitories": 30,
    "anchors": 30,
    "parking": 30,
}


def _dedup(gdf, radius_m):
    """Greedy de-duplication: drop points within radius_m of an already kept one (node+polygon double mapping)."""
    xy = np.radians(np.c_[gdf.geometry.y.values, gdf.geometry.x.values])
    tree = BallTree(xy, metric="haversine")
    nb = tree.query_radius(xy, r=radius_m / R_EARTH)
    keep = np.ones(len(gdf), bool)
    for i in range(len(gdf)):
        if keep[i]:
            for j in nb[i]:
                if j > i:
                    keep[j] = False
    return gdf[keep].reset_index(drop=True)


class _PointLayer:
    def __init__(self, name):
        g = gpd.read_parquet(OSM + name + ".parquet")
        if name in DEDUP_M:
            g = _dedup(g, DEDUP_M[name])
        self.n = len(g)
        self.tree = BallTree(np.radians(np.c_[g.geometry.y.values, g.geometry.x.values]), metric="haversine")

    def count(self, lat, lon, r_m):
        return int(self.tree.query_radius(np.radians([[lat, lon]]), r=r_m / R_EARTH, count_only=True)[0])

    def nearest(self, lat, lon):
        d, _ = self.tree.query(np.radians([[lat, lon]]), k=1)
        return float(d[0, 0] * R_EARTH)


class _ShapeLayer:
    """Polygons / lines / points kept as true geometry; distance = to the shape (0 if inside a polygon)."""

    def __init__(self, name):
        g = gpd.read_parquet(OSM + name + ".parquet")
        self.geoms = np.array(g.geometry.values, dtype=object)
        self.tree = STRtree(self.geoms)

    def nearest_m(self, lat, lon, max_deg=6.0):
        proj = _local_proj(lat, lon)
        p = Point(lon, lat)
        w = 0.02
        while w <= max_deg:
            idx = self.tree.query(p.buffer(w))
            if len(idx):
                ds = [shp_transform(proj, self.geoms[i]).distance(Point(0, 0)) for i in idx]
                d = min(ds)
                if d <= w * 75000:  # 1 deg lon >= ~75 km at latitudes < 47: window fully covers radius d
                    return float(d)
            w *= 3
        return float("nan")


class _RoadLayer:
    """Road lines with a highway class: nearest distance to a class subset and total length in a radius."""

    def __init__(self):
        self.g = gpd.read_parquet(OSM + "road_classes.parquet").reset_index(drop=True)
        self.tree = STRtree(np.array(self.g.geometry.values, dtype=object))
        self.hw = self.g["highway"].to_numpy()

    def _near(self, lat, lon, w):
        return self.tree.query(Point(lon, lat).buffer(w))

    def nearest_m(self, lat, lon, classes, max_deg=6.0):
        proj = _local_proj(lat, lon)
        w = 0.02
        while w <= max_deg:
            idx = [i for i in self._near(lat, lon, w) if self.hw[i] in classes]
            if idx:
                d = min(shp_transform(proj, self.g.geometry.iloc[i]).distance(Point(0, 0)) for i in idx)
                if d <= w * 75000:
                    return float(d)
            w *= 3
        return float("nan")

    def length_m(self, lat, lon, classes, r_m):
        proj = _local_proj(lat, lon)
        disc = Point(0, 0).buffer(r_m, 64)
        total = 0.0
        for i in self._near(lat, lon, r_m / 111000.0 * 1.6 + 0.001):
            if self.hw[i] in classes:
                total += shp_transform(proj, self.g.geometry.iloc[i]).intersection(disc).length
        return float(total)


class _LineLayer:
    """Plain line layer; total length (m) inside a radius, local azimuthal-equidistant projection."""

    def __init__(self, name):
        g = gpd.read_parquet(OSM + name + ".parquet")
        self.geoms = np.array(g.geometry.values, dtype=object)
        self.tree = STRtree(self.geoms)

    def length_m(self, lat, lon, r_m):
        proj = _local_proj(lat, lon)
        disc = Point(0, 0).buffer(r_m, 64)
        total = 0.0
        for i in self.tree.query(Point(lon, lat).buffer(r_m / 111000.0 * 1.6 + 0.001)):
            total += shp_transform(proj, self.geoms[i]).intersection(disc).length
        return float(total)


class _StreetLayer:
    """Corner proxy: named car streets (name, else ref) and the junction nodes where >=2 distinct streets meet."""

    def __init__(self):
        g = gpd.read_parquet(OSM + "street_lines.parquet")
        self.geoms = np.array(g.geometry.values, dtype=object)
        self.keys = g["key"].to_numpy()
        self.tree = STRtree(self.geoms)
        j = gpd.read_parquet(OSM + "street_junctions.parquet")
        self.jkeys = [set(k.split("\x1f")) for k in j["keys"]]
        self.jtree = BallTree(np.radians(np.c_[j.geometry.y.values, j.geometry.x.values]), metric="haversine")

    def corner(self, lat, lon, street_m=40, junction_m=60):
        proj = _local_proj(lat, lon)
        origin = Point(0, 0)
        streets = {
            self.keys[i]
            for i in self.tree.query(Point(lon, lat).buffer(0.001))
            if shp_transform(proj, self.geoms[i]).distance(origin) <= street_m
        }
        if len(streets) < 2:
            return 0
        for ix in self.jtree.query_radius(np.radians([[lat, lon]]), r=junction_m / R_EARTH)[0]:
            if len(self.jkeys[ix] & streets) >= 2:
                return 1
        return 0


def _local_proj(lat, lon):
    t = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m", always_xy=True)
    return t.transform


@lru_cache(None)
def _points(name):
    return _PointLayer(name)


@lru_cache(None)
def _shapes(name):
    return _ShapeLayer(name)


@lru_cache(None)
def _roads():
    return _RoadLayer()


@lru_cache(None)
def _lines(name):
    return _LineLayer(name)


@lru_cache(None)
def _streets():
    return _StreetLayer()


@lru_cache(None)
def _tashkent_poly():
    return gpd.read_file(OSM + "tashkent_city.geojson").geometry.iloc[0]


@lru_cache(None)
def _raster():
    return rasterio.open(POP_TIF)


def _hav(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def population_within(lat, lon, radii=(300, 1000, 5000, 10000)):
    """Sum of WorldPop cells whose CENTRE lies within each radius (haversine). NaN if outside raster."""
    r = _raster()
    rmax = max(radii)
    dlat = rmax / 111000.0 + 0.002
    dlon = dlat / np.cos(np.radians(lat))
    row0, col0 = r.index(lon - dlon, lat + dlat)
    row1, col1 = r.index(lon + dlon, lat - dlat)
    if row1 < 0 or col1 < 0 or row0 >= r.height or col0 >= r.width:
        return {rad: float("nan") for rad in radii}
    row0, col0 = max(row0, 0), max(col0, 0)
    row1, col1 = min(row1, r.height - 1), min(col1, r.width - 1)
    win = Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1)
    a = r.read(1, window=win).astype("float64")
    a[(a == r.nodata) | ~np.isfinite(a) | (a < 0)] = 0.0
    rows, cols = np.mgrid[row0 : row1 + 1, col0 : col1 + 1]
    xs, ys = rasterio.transform.xy(r.transform, rows.ravel(), cols.ravel(), offset="center")
    d = _hav(lat, lon, np.array(ys), np.array(xs)).reshape(a.shape)
    return {rad: float(a[d <= rad].sum()) for rad in radii}


@lru_cache(None)
def _ghsl_tiles():
    tiles = [rasterio.open(p) for p in sorted(glob.glob(GHSL_DIR + "*.tif"))]
    if not tiles:
        raise FileNotFoundError("no GHS-BUILT-V tiles in " + GHSL_DIR)
    return (
        tiles,
        Transformer.from_crs("EPSG:4326", tiles[0].crs, always_xy=True),
        Transformer.from_crs(tiles[0].crs, "EPSG:4326", always_xy=True),
    )


def built_volume_within(lat, lon, r_m=500):
    """Sum of GHS-BUILT-V (m3 per 100 m cell) over cells whose CENTRE lies within r_m (haversine on the
    cell-centre lon/lat) of the point. Windowed read of the Mollweide tiles; NaN if the point is outside all tiles."""
    tiles, to_moll, from_moll = _ghsl_tiles()
    x, y = to_moll.transform(lon, lat)
    pad = r_m * 1.6 + 200  # Mollweide scale distortion at <= 46 N is < 1.3; pad generously, haversine does the cut
    covered, total = False, 0.0
    for t in tiles:
        b = t.bounds
        if x + pad < b.left or x - pad > b.right or y + pad < b.bottom or y - pad > b.top:
            continue
        win = (
            rasterio.windows.from_bounds(
                max(x - pad, b.left),
                max(y - pad, b.bottom),
                min(x + pad, b.right),
                min(y + pad, b.top),
                transform=t.transform,
            )
            .round_offsets()
            .round_lengths()
        )
        a = t.read(1, window=win).astype("float64")
        a[(a == t.nodata) | ~np.isfinite(a) | (a < 0)] = 0.0
        rows, cols = np.mgrid[
            int(win.row_off) : int(win.row_off) + a.shape[0], int(win.col_off) : int(win.col_off) + a.shape[1]
        ]
        xs, ys = rasterio.transform.xy(t.transform, rows.ravel(), cols.ravel(), offset="center")
        lo, la = from_moll.transform(np.array(xs), np.array(ys))
        d = _hav(lat, lon, la, lo).reshape(a.shape)
        total += float(a[d <= r_m].sum())
        covered = True
    return total if covered else float("nan")


@lru_cache(None)
def _nl_raster():
    return rasterio.open(NL_TIF)


def night_lights_within(lat, lon, r_m=500, return_info=False):
    """Mean radiance (nW/cm2/sr = raw/NL_SCALE) of the ~500 m VIIRS cells whose CENTRE lies within r_m (haversine)
    of the point; if no valid cell centre is within r_m, the single cell containing the point (fallback).
    NaN if the point is outside the raster / only nodata. return_info=True -> (value, used_fallback)."""
    r = _nl_raster()
    dlat = r_m / 111000.0 + 0.01
    dlon = dlat / np.cos(np.radians(lat))
    row0, col0 = r.index(lon - dlon, lat + dlat)
    row1, col1 = r.index(lon + dlon, lat - dlat)
    row0, col0 = max(row0, 0), max(col0, 0)
    row1, col1 = min(row1, r.height - 1), min(col1, r.width - 1)
    val, fb = float("nan"), False
    if row1 >= row0 and col1 >= col0:
        a = r.read(1, window=Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1)).astype("float64")
        ok = (a != r.nodata) & np.isfinite(a)
        rows, cols = np.mgrid[row0 : row1 + 1, col0 : col1 + 1]
        xs, ys = rasterio.transform.xy(r.transform, rows.ravel(), cols.ravel(), offset="center")
        d = _hav(lat, lon, np.array(ys), np.array(xs)).reshape(a.shape)
        m = (d <= r_m) & ok
        if m.any():
            val = float(a[m].mean() / NL_SCALE)
        else:
            fb = True
            ri, ci = r.index(lon, lat)
            if 0 <= ri < r.height and 0 <= ci < r.width:
                v = float(r.read(1, window=Window(ci, ri, 1, 1))[0, 0])
                val = v / NL_SCALE if v != r.nodata else float("nan")
    return (val, fb) if return_info else val


def load_safia(path=DEFAULT_SAFIA):
    """Safia locations: needs lat, lon and a code column ('code' or 'matched_code'); other columns ignored."""
    df = pd.read_csv(path)
    code = df["code"] if "code" in df.columns else df.get("matched_code", pd.Series([None] * len(df)))
    return pd.DataFrame(
        {"code": code.values, "lat": df["lat"].astype(float).values, "lon": df["lon"].astype(float).values}
    )


def compute_features(lat, lon, exclude_code=None, safia=None, safia_known=None):
    """Return dict of features for (lat, lon). `safia` = DataFrame from load_safia (default: branches.csv)."""
    lat, lon = float(lat), float(lon)
    f = {}
    pop = population_within(lat, lon)
    f["pop_300m"], f["pop_1km"], f["pop_5km"], f["pop_10km"] = pop[300], pop[1000], pop[5000], pop[10000]
    P = _points
    f["schools_1km"] = P("schools").count(lat, lon, 1000)
    f["kindergartens_1km"] = P("kindergartens").count(lat, lon, 1000)
    f["universities_1km"] = P("universities").count(lat, lon, 1000)
    f["offices_1km"] = P("offices").count(lat, lon, 1000)
    f["hospitals_1km"] = P("hospitals").count(lat, lon, 1000)
    f["dist_metro_m"] = P("metro").nearest(lat, lon)
    f["metro_within_800m"] = int(f["dist_metro_m"] <= 800)
    f["bus_stops_300m"] = P("bus_stops").count(lat, lon, 300)
    f["dist_bazaar_m"] = _shapes("bazaars").nearest_m(lat, lon)
    f["dist_mall_m"] = _shapes("malls").nearest_m(lat, lon)
    f["in_mall"] = int(f["dist_mall_m"] <= 50)
    f["dist_main_road_m"] = _shapes("main_roads").nearest_m(lat, lon)
    f["food_places_300m"] = P("food").count(lat, lon, 300)
    f["shops_300m"] = P("shops").count(lat, lon, 300)
    f["bakeries_1km"] = P("bakeries").count(lat, lon, 1000)
    # other Safia stores: full network (safia_network.csv) and the 127 known branches only
    for stores, dist_name, count_name in (
        (safia, "dist_nearest_safia_m", "safia_within_1km"),
        (safia_known, "dist_nearest_safia_known_m", "safia_known_within_1km"),
    ):
        if stores is None:
            stores = load_safia(DEFAULT_SAFIA if dist_name == "dist_nearest_safia_m" else KNOWN_SAFIA)
        if exclude_code is not None:
            stores = stores[stores["code"].astype(str) != str(exclude_code)]
        if len(stores):
            d = _hav(lat, lon, stores["lat"].values, stores["lon"].values)
            f[dist_name], f[count_name] = float(d.min()), int((d <= 1000).sum())
        else:
            f[dist_name], f[count_name] = float("nan"), 0
    # roads, dormitories, anchors
    R = _roads()
    f["dist_major_road_m"] = R.nearest_m(lat, lon, {"trunk", "primary", "trunk_link", "primary_link"})
    f["major_intersection_300m"] = int(P("major_intersections").count(lat, lon, 300) > 0)
    f["major_road_len_500m"] = R.length_m(lat, lon, {"trunk", "primary", "secondary"}, 500)
    f["dormitories_1km"] = P("dormitories").count(lat, lon, 1000)
    f["universities_500m"] = P("universities").count(lat, lon, 500)
    f["anchors_300m"] = P("anchors").count(lat, lon, 300)
    # city
    f["is_tashkent"] = int(_tashkent_poly().contains(Point(lon, lat)))
    # parking, footpaths, crossings, street proxies
    f["parking_300m"] = P("parking").count(lat, lon, 300)
    f["dist_parking_m"] = _shapes("parking_shapes").nearest_m(lat, lon)
    f["footpath_len_500m"] = _lines("footpaths").length_m(lat, lon, 500)
    f["walk_intersections_500m"] = P("walk_intersections").count(lat, lon, 500)
    f["crossings_300m"] = P("crossings").count(lat, lon, 300)
    f["on_main_street"] = int(R.nearest_m(lat, lon, {"trunk", "primary", "secondary"}) <= 30)  # proxy
    f["corner_site"] = _streets().corner(lat, lon)  # proxy
    # raster features
    f["built_volume_500m"] = built_volume_within(lat, lon, 500)
    f["night_lights_500m"] = night_lights_within(lat, lon, 500)
    return f
