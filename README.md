# Safia location sales index

Predicts a 1-100 sales index for a proposed Safia location from its surroundings (OpenStreetMap, population, built volume, night lights).

## Setup
```
conda env create -f environment.yml
conda activate safia
```

## Usage
```
python predict.py --lat 41.31 --lon 69.28
```
- `--hours 24/7|day`: opening hours you plan for the store.
- `--age-months N`: store age in months (default: training median).
- `--store-code CODE` / `--new-site`: if an existing Safia store lies within 30 m of the point it is treated as that store (excluded from its own neighbour features); `--store-code` forces this, `--new-site` forces new-site treatment.
- `--json`: machine-readable output.
- `--csv in.csv --out out.csv`: batch mode. Required columns `lat`, `lon`; optional `hours`, `age_months`, `store_code`, `new_site`.

Output: the predicted sales index (1-100). Use `--json` for the full details: 80% range, store inputs used, location features, similar branches and warnings.

Example (`python predict.py --lat 41.31 --lon 69.28`):
```
Sales index: 15.5
```

## Test
```
python tests/test_predict.py
```

## Files
```
predict.py              command-line entry point
safia_model/            package: features, data, models, evaluate, intervals, predict_lib
models/final_m6d.joblib the shipped model (only m6d)
tests/test_predict.py   smoke test
data/processed/         branches.csv, safia_network.csv, store_attributes.csv,
                        osm/ (24 parquet layers + tashkent_city.geojson)
data/raw/               WorldPop population, VIIRS night lights, 6 GHSL built-volume tiles
environment.yml         conda environment (name: safia)
```

## Data sources
- OpenStreetMap, Geofabrik extract of 29 Sep 2026 (ODbL).
- WorldPop 2026, 100 m, constrained, release R2025A v1 (CC BY 4.0).
- EU JRC GHSL GHS-BUILT-V 2025 (R2023A, 100 m), six tiles (CC BY 4.0).
- VIIRS 2024 annual night lights via the OpenGeoHub Zenodo copy, record 17294744 (CC BY 4.0); a third-party mirror, not the official EOG file.
- Safia's website: store list, opening hours and archived listings (Wayback Machine).
