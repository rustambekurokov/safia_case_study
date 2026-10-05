import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(lat, lon, *flags):
    cmd = [sys.executable, os.path.join(ROOT, "predict.py"), "--lat", str(lat), "--lon", str(lon), "--json", *flags]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def test_points():
    c = run(41.31, 69.28)
    assert 1 <= c["prediction"] <= 100 and c["lo"] < c["prediction"] < c["hi"]
    r = run(40.10, 67.84)
    assert r["prediction"] is not None and r["warnings"]
    o = run(55.75, 37.60)
    assert o["prediction"] is None and "OUTSIDE UZBEKISTAN" in " ".join(o["warnings"])


def test_existing_store_vs_new_site():
    e = run(41.3275, 69.3257)  # ~5 m from N001: treated as N001
    assert e["site"]["mode"] == "existing" and e["site"]["code"] == "N001" and e["site"]["dist_m"] < 30
    assert abs(e["features"]["dist_nearest_safia_m"] - 248) < 5
    assert not any("extrapolation" in w for w in e["warnings"])
    assert abs(e["prediction"] - run(41.3275, 69.3257, "--store-code", "N001")["prediction"]) < 1e-9

    n = run(41.3285378, 69.3256844)  # ~120 m from N001: new site, N001 is a neighbour
    assert n["site"]["mode"] == "new" and n["site"]["nearest"]["code"] == "N001"
    assert 100 < n["site"]["nearest"]["dist_m"] < 140
    assert 100 < n["features"]["dist_nearest_safia_m"] < 140
    assert all("DEFAULT" in v for v in n["store_inputs"]["sources"].values())  # default hours/age

    f = run(41.3275, 69.3257, "--new-site")  # forced new site at N001: it counts as a neighbour
    assert f["site"]["mode"] == "new" and f["site"]["nearest"]["code"] == "N001"
    assert f["features"]["dist_nearest_safia_m"] < 10


if __name__ == "__main__":
    test_points()
    test_existing_store_vs_new_site()
    print("ok")
