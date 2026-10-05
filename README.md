# Safia location sales index

Predicts a 1-100 sales index for a proposed Safia location from its surroundings (OpenStreetMap, population, built volume, night lights). Method and results are in `Safia_case_study_writeup.pdf` (also `.docx`).

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

Output: the predicted index, the 80% range, the store inputs used (hours, age and where each came from), the location features, the most similar training branches, and any warnings (for example a point far from all training branches).

Example (`python predict.py --lat 41.31 --lon 69.28`):
```
Location: lat 41.31000, lon 69.28000
Site: treated as a new site (no Safia store within 30 m); nearest Safia N030 Пойтахт is 515 m away and counts as a neighbour
Predicted sales index: 15.5  (scale 1-100; model: M6d Blend of M1d and coordinates kNN k=10)
80% prediction interval: 9.1 - 26.3  (prediction x/÷ 1.70; split-conformal from out-of-fold grouped-CV residuals)
  The interval is wide because the model still explains only part of the variation between branches (cross-validated R2 in log space is roughly 0.3 for this model, on branches with known 24/7 and age). open_24_7 and age are STORE DECISIONS / history that you set, not properties of the location; the prediction is conditional on them. High sellers are still under-predicted.
Store inputs (STORE DECISIONS / history, not location properties; the prediction is conditional on them):
  open_24_7 = 0 (not 24/7) <- DEFAULT: most common value among training branches (new site)
  age_months = 37.1 <- DEFAULT: median age_months among training branches (new site)
Features:
  dist_nearest_safia_m=514.794; shops_300m=1; food_places_300m=6; bakeries_1km=5; pop_1km=31838.7; pop_5km=699766; kindergartens_1km=4; offices_1km=88; bus_stops_300m=1; metro_within_800m=1; in_mall=0; is_tashkent=1; open_24_7=0; log_age_months=3.64146; night_lights_500m=7.2; built_volume_500m=3.5689e+06
Most similar training branches (standardised feature space; sales not shown):
  W172 Заркайнар (Tashkent): feature distance 3.32, 3.7 km away
  W182 Дружба (Tashkent): feature distance 3.71, 3.0 km away
  W185 Бульвар (Tashkent): feature distance 3.77, 3.0 km away
Warnings: none
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
Safia_case_study_writeup.pdf / .docx   write-up
```

## Data sources
- OpenStreetMap, Geofabrik extract of 29 Sep 2026 (ODbL).
- WorldPop 2026, 100 m, constrained, release R2025A v1 (CC BY 4.0).
- EU JRC GHSL GHS-BUILT-V 2025 (R2023A, 100 m), six tiles (CC BY 4.0).
- VIIRS 2024 annual night lights via the OpenGeoHub Zenodo copy, record 17294744 (CC BY 4.0); a third-party mirror, not the official EOG file.
- Safia's website: store list, opening hours and archived listings (Wayback Machine).
