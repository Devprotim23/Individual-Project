from __future__ import annotations

"""
models.py

This module contains the forecasting methods used by the project.

Active final methods:
- Seasonal naïve
- ETS

ARIMA is intentionally retained in this file because it was explored during the
project, but it is not part of the final active submitted scope.
"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd


class ModelError(Exception):
    """Raised when a forecasting model cannot be fit or used safely."""


# Validate the common forecast inputs used by all model functions.
def _validate_forecast_inputs(series: pd.Series, horizon: int) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ModelError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ModelError("Series index must be a pandas DatetimeIndex.")

    if horizon <= 0:
        raise ModelError("Forecast horizon h must be a positive integer.")

    series = series.sort_index().asfreq("MS")

    if series.isna().any():
        raise ModelError("Series contains missing values; clean or impute before forecasting.")

    return series.astype(float)


# Build the future monthly index shared by all forecast methods.
def _build_future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    last_date = series.index.max()

    return pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )


# Build the standard forecast output DataFrame used throughout the project.
def _build_forecast_output(
    future_index: pd.DatetimeIndex,
    method_name: str,
    point_forecast: np.ndarray,
    lo80: np.ndarray,
    hi80: np.ndarray,
    lo95: np.ndarray,
    hi95: np.ndarray,
) -> pd.DataFrame:
    horizon = len(future_index)

    return pd.DataFrame(
        {
            "date": future_index,
            "method": method_name,
            "horizon": np.arange(1, horizon + 1, dtype=int),
            "point": np.asarray(point_forecast, dtype=float),
            "lo80": np.asarray(lo80, dtype=float),
            "hi80": np.asarray(hi80, dtype=float),
            "lo95": np.asarray(lo95, dtype=float),
            "hi95": np.asarray(hi95, dtype=float),
        }
    )


# One-step seasonal naive residuals for a monthly series:
# residual_t = actual_t - actual_(t-12)
def _seasonal_naive_residuals(series: pd.Series) -> pd.Series:
    fitted = series.shift(12)
    residuals = (series - fitted).dropna()
    return residuals.astype(float)


# Seasonal naive forecast for a monthly series.
#
# For monthly data, each forecast is the value from the same month
# one year earlier.
#
# Prediction intervals use empirical quantiles of historical
# seasonal-naive residuals.
def forecast_seasonal_naive(series: pd.Series, h: int) -> pd.DataFrame:
    series = _validate_forecast_inputs(series, h)

    # Seasonal naïve intervals need enough history to compute seasonal residuals.
    if len(series) < 24:
        raise ModelError(
            "Seasonal naive forecasting with intervals requires at least 24 observations."
        )

    future_index = _build_future_index(series, h)

    points: list[float] = []

    # For each future month, copy the value from the same month one year earlier.
    for future_date in future_index:
        seasonal_lag_date = future_date - pd.DateOffset(years=1)

        if seasonal_lag_date not in series.index:
            raise ModelError(
                f"Missing seasonal lag value for forecast date {future_date.date()} "
                f"(expected {seasonal_lag_date.date()})."
            )

        points.append(float(series.loc[seasonal_lag_date]))

    residuals = _seasonal_naive_residuals(series)

    if len(residuals) < 12:
        raise ModelError("Not enough historical residuals to estimate prediction intervals.")

    # Use empirical residual quantiles to form simple 80% and 95% intervals.
    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    q025, q975 = np.quantile(residuals, [0.025, 0.975])

    points = np.asarray(points, dtype=float)

    return _build_forecast_output(
        future_index=future_index,
        method_name="seasonal_naive",
        point_forecast=points,
        lo80=points + q10,
        hi80=points + q90,
        lo95=points + q025,
        hi95=points + q975,
    )


# ETS forecast for a monthly series using additive trend and additive seasonality.
def forecast_ets(series: pd.Series, h: int) -> pd.DataFrame:
    series = _validate_forecast_inputs(series, h)

    if len(series) < 24:
        raise ModelError("ETS forecasting requires at least 24 observations.")

    future_index = _build_future_index(series, h)

    try:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
    except Exception as error:
        raise ModelError(f"ETS model fitting failed: {error}") from error

    point = np.asarray(fit.forecast(h), dtype=float)

    residuals = np.asarray(fit.resid, dtype=float)
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < 12:
        raise ModelError("Not enough ETS residuals to estimate prediction intervals.")

    # Use empirical residual quantiles to create simple forecast intervals.
    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    q025, q975 = np.quantile(residuals, [0.025, 0.975])

    return _build_forecast_output(
        future_index=future_index,
        method_name="ets",
        point_forecast=point,
        lo80=point + q10,
        hi80=point + q90,
        lo95=point + q025,
        hi95=point + q975,
    )


# Seasonal ARIMA forecast for a monthly series.
#
# Uses a simple SARIMA specification suitable as an MVP.
#
# This function is intentionally retained for explanation and evidence of
# attempted development, but ARIMA is not part of the final active scope.
def forecast_arima(series: pd.Series, h: int) -> pd.DataFrame:
    series = _validate_forecast_inputs(series, h)

    # ARIMA needs a longer history because it includes both regular and seasonal differencing.
    if len(series) < 36:
        raise ModelError("ARIMA forecasting requires at least 36 observations.")

    try:
        model = SARIMAX(
            series,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
    except Exception as error:
        raise ModelError(f"ARIMA model fitting failed: {error}") from error

    try:
        forecast_result = fit.get_forecast(steps=h)
        mean_forecast = np.asarray(forecast_result.predicted_mean, dtype=float)

        ci80 = forecast_result.conf_int(alpha=0.20)
        ci95 = forecast_result.conf_int(alpha=0.05)
    except Exception as error:
        raise ModelError(f"ARIMA forecasting failed: {error}") from error

    future_index = _build_future_index(series, h)

    # conf_int may be returned as either a DataFrame or a NumPy array,
    # so handle both safely.
    if isinstance(ci80, pd.DataFrame):
        lo80 = ci80.iloc[:, 0].to_numpy(dtype=float)
        hi80 = ci80.iloc[:, 1].to_numpy(dtype=float)
    else:
        lo80 = np.asarray(ci80[:, 0], dtype=float)
        hi80 = np.asarray(ci80[:, 1], dtype=float)

    if isinstance(ci95, pd.DataFrame):
        lo95 = ci95.iloc[:, 0].to_numpy(dtype=float)
        hi95 = ci95.iloc[:, 1].to_numpy(dtype=float)
    else:
        lo95 = np.asarray(ci95[:, 0], dtype=float)
        hi95 = np.asarray(ci95[:, 1], dtype=float)

    return _build_forecast_output(
        future_index=future_index,
        method_name="arima",
        point_forecast=mean_forecast,
        lo80=lo80,
        hi80=hi80,
        lo95=lo95,
        hi95=hi95,
    )