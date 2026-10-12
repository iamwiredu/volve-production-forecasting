# Volve Production Forecasting: Machine Learning vs Arps Decline Curves

**Status: in progress**

How well can machine learning forecast oil production compared with classical Arps decline-curve analysis, and where does each one fail? This project uses real field data from Equinor's **Volve** field in the Norwegian North Sea.

## Why this question
Arps decline curves assume steady, boundary-dominated flow and fixed operating conditions. Real wells see choke changes, shut-ins, water breakthrough and pressure support from injection. ML models can use these operating variables, but they ignore physics and can extrapolate badly. This project measures both approaches on the same wells and the same held-out period.

## Plan
| Stage | Content | Status |
|---|---|---|
| 1 | Load and clean daily production data; monthly rates per well | ✅ Code ready |
| 2 | Arps hyperbolic baseline, trained on the first 80% of history and tested on the last 20% | ✅ Code and tests ready; running on the data |
| 3 | ML forecasts (gradient boosting, then LSTM) using lagged rates, on-stream hours, choke size, and wellhead and downhole pressure | ⏳ Next |
| 4 | Comparison: RMSE, MAPE and cumulative-volume error per well, plus analysis of where each method fails | ⏳ |
| 5 | Hybrid: use the Arps fit as a feature or a physics constraint for the ML model | ⏳ |
| 6 | Write-up | ⏳ |

## Repository layout
```
src/volve/data.py      load and clean the Volve daily production data
src/volve/arps.py      Arps hyperbolic decline: fitting and forecasting
src/volve/metrics.py   RMSE, MAPE and cumulative-volume error
notebooks/01_explore_and_arps_baseline.ipynb
tests/                 checks that the Arps fit recovers known parameters
data/                  place the Volve data here (not committed)
```

## Run it
```bash
pip install -r requirements.txt
pytest
jupyter notebook notebooks/
```
Download the data first. See [data/README.md](data/README.md).

## Data
Volve production data © Equinor, released under the Equinor Open Data Licence. The data is not redistributed in this repository.

## Author
Kwaku Appenteng Wiredu, BSc Petroleum Engineering, KNUST
