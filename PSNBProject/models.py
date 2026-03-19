from __future__ import annotations
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np
import pandas as pd


class ModelError(Exception):
    """Raised when a forecasting model cannot be fit or used safely."""


def _seasonal_naive_residuals(s: pd.Series) -> pd.Series:
    """
    One-step seasonal naive residuals for a monthly series:
    residual_t = actual_t - actual_(t-12)
    """
    fitted = s.shift(12)
    residuals = (s - fitted).dropna()
    return residuals.astype(float)


def forecast_seasonal_naive(s: pd.Series, h: int) -> pd.DataFrame:
    """
    Seasonal naive forecast for a monthly series.

    For monthly data, each forecast is the value from the same month
    one year earlier.

    Prediction intervals use empirical quantiles of historical
    seasonal-naive residuals.
    """
    if not isinstance(s, pd.Series):
        raise ModelError("Input must be a pandas Series.")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise ModelError("Series index must be a pandas DatetimeIndex.")

    if h <= 0:
        raise ModelError("Forecast horizon h must be a positive integer.")

    s = s.sort_index().asfreq("MS")

    if len(s) < 24:
        raise ModelError(
            "Seasonal naive forecasting with intervals requires at least 24 observations."
        )

    if s.isna().any():
        raise ModelError("Series contains missing values; clean or impute before forecasting.")

    last_date = s.index.max()
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=h,
        freq="MS",
    )

    points = []
    for d in future_index:
        seasonal_lag_date = d - pd.DateOffset(years=1)
        if seasonal_lag_date not in s.index:
            raise ModelError(
                f"Missing seasonal lag value for forecast date {d.date()} "
                f"(expected {seasonal_lag_date.date()})."
            )
        points.append(float(s.loc[seasonal_lag_date]))

    residuals = _seasonal_naive_residuals(s)

    if len(residuals) < 12:
        raise ModelError("Not enough historical residuals to estimate prediction intervals.")

    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    q025, q975 = np.quantile(residuals, [0.025, 0.975])

    points = np.asarray(points, dtype=float)

    df = pd.DataFrame(
        {
            "date": future_index,
            "method": "seasonal_naive",
            "horizon": np.arange(1, h + 1, dtype=int),
            "point": points,
            "lo80": points + q10,
            "hi80": points + q90,
            "lo95": points + q025,
            "hi95": points + q975,
        }
    )

    return df


def forecast_ets(s: pd.Series, h: int) -> pd.DataFrame:
    """
    ETS forecast for a monthly series using additive trend and additive seasonality.
    """
    if not isinstance(s, pd.Series):
        raise ModelError("Input must be a pandas Series.")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise ModelError("Series index must be a pandas DatetimeIndex.")

    if h <= 0:
        raise ModelError("Forecast horizon h must be a positive integer.")

    s = s.sort_index().asfreq("MS")

    if len(s) < 24:
        raise ModelError("ETS forecasting requires at least 24 observations.")

    if s.isna().any():
        raise ModelError("Series contains missing values; clean or impute before forecasting.")

    last_date = s.index.max()
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=h,
        freq="MS",
    )

    try:
        model = ExponentialSmoothing(
            s.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
    except Exception as e:
        raise ModelError(f"ETS model fitting failed: {e}") from e

    point = np.asarray(fit.forecast(h), dtype=float)

    residuals = np.asarray(fit.resid, dtype=float)
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < 12:
        raise ModelError("Not enough ETS residuals to estimate prediction intervals.")

    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    q025, q975 = np.quantile(residuals, [0.025, 0.975])

    df = pd.DataFrame(
        {
            "date": future_index,
            "method": "ets",
            "horizon": np.arange(1, h + 1, dtype=int),
            "point": point,
            "lo80": point + q10,
            "hi80": point + q90,
            "lo95": point + q025,
            "hi95": point + q975,
        }
    )

    return df

def forecast_arima(s: pd.Series, h: int) -> pd.DataFrame:
    """
    Seasonal ARIMA forecast for a monthly series.

    Uses a simple SARIMA specification suitable as an MVP:
    order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)

    Returns
    -------
    pd.DataFrame
        Columns:
        date, method, horizon, point, lo80, hi80, lo95, hi95
    """
    if not isinstance(s, pd.Series):
        raise ModelError("Input must be a pandas Series.")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise ModelError("Series index must be a pandas DatetimeIndex.")

    if h <= 0:
        raise ModelError("Forecast horizon h must be a positive integer.")

    s = s.sort_index().asfreq("MS")

    if len(s) < 36:
        raise ModelError("ARIMA forecasting requires at least 36 observations.")

    if s.isna().any():
        raise ModelError("Series contains missing values; clean or impute before forecasting.")

    try:
        model = SARIMAX(
            s.astype(float),
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
    except Exception as e:
        raise ModelError(f"ARIMA model fitting failed: {e}") from e

    try:
        pred80 = fit.get_forecast(steps=h)
        mean_forecast = np.asarray(pred80.predicted_mean, dtype=float)

        ci80 = pred80.conf_int(alpha=0.20)
        ci95 = pred80.conf_int(alpha=0.05)
    except Exception as e:
        raise ModelError(f"ARIMA forecasting failed: {e}") from e

    last_date = s.index.max()
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=h,
        freq="MS",
    )

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

    df = pd.DataFrame(
        {
            "date": future_index,
            "method": "arima",
            "horizon": np.arange(1, h + 1, dtype=int),
            "point": mean_forecast,
            "lo80": lo80,
            "hi80": hi80,
            "lo95": lo95,
            "hi95": hi95,
        }
    )

    return df