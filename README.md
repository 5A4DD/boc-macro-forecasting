# Canadian Macroeconomic Forecasting with XGBoost

XGBoost models forecasting Canadian real GDP growth, total CPI inflation and core CPI inflation. The project was developed for Carleton University's submission to the Bank of Canada Governor's Challenge 2025. Forecasts from these models were ensembled with VAR model forecasts to inform the team's final policy recommendation.

## Models

Each notebook forecasts one target on a year-over-year basis.

| Notebook | Target | Frequency |
|---|---|---|
| `GDP_YoY.ipynb` | Real GDP growth | Quarterly |
| `CPI_YoY.ipynb` | Total CPI inflation | Monthly |
| `CPI_Core_YoY.ipynb` | Core CPI inflation | Monthly |

## Repository Structure

```
GDP_YoY.ipynb             Quarterly GDP model
CPI_YoY.ipynb             Monthly total CPI model
CPI_Core_YoY.ipynb        Monthly core CPI model
Quarterly_data_new.csv    Quarterly input data
Monthly_data_new.csv      Monthly input data
GDP_Results.csv           GDP forecasts
CPI_Results.csv           Total CPI forecasts
Core_CPI_Results.csv      Core CPI forecasts
```

## Methodology

**Inputs.** BoC target rate, WTI and Brent crude prices, CAD/USD real exchange rate, US real GDP growth, Canadian unemployment rate, population growth, and GDP expenditure components (private consumption, government spending, gross investment, exports, imports). Most series are transformed to year-over-year percent changes.

**Features.** Up to 12 lags for monthly models and 4 lags for the quarterly model, along with a month or quarter indicator and a linear time trend.

**Model.** `XGBRegressor` with `n_estimators=1000`, `max_depth=5`, `learning_rate=0.1` and `subsample=0.7`.

**Validation.** 5-fold `TimeSeriesSplit` cross-validation. MAE, MSE, RMSE and R² are reported for each fold and on average, with plots of actual against predicted values.

**Forecasting.** All variables are predicted jointly with `MultiOutputRegressor`, and each step's predictions are used as lagged inputs for the next step. The GDP model forecasts 8 quarters ahead and the CPI models forecast up to 24 months ahead.

## Forecast Assumptions

The forward forecasts use the following assumptions for exogenous variables:

- BoC target rate 25 bps below its last observed value
- WTI at $60/barrel and Brent at $65/barrel
- Real exchange rate held at its last observed value
- Population growth held at 0.5% YoY

These values are set in `forecast_multivar_single_model()` and can be modified to run alternative scenarios.

## Requirements

Python 3 with `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib` and `seaborn`. The notebooks read input data from the repository root.

## Data Sources

Publicly available data from the Bank of Canada, Statistics Canada and financial market sources.
