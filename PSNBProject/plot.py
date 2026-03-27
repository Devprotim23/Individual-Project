from __future__ import annotations

"""
plot.py

This module creates exported forecast charts showing:
- recent historical values
- a single forecast path
- 80% and 95% prediction intervals
- a provenance/disclaimer footer for the saved PNG
"""

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import pandas as pd


# Raised when a forecast plot cannot be created safely.
class PlotError(Exception):
    pass


# Create and save a forecast plot for a single forecasting method.
def plot_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "PSNB Forecast",
    ylabel: str = "PSNB",
    history_tail: Optional[int] = 60,
    meta: Optional[Mapping[str, str]] = None,
    method_label: Optional[str] = None,
) -> Path:
    if not isinstance(history, pd.Series):
        raise PlotError("history must be a pandas Series.")

    if not isinstance(history.index, pd.DatetimeIndex):
        raise PlotError("history index must be a pandas DatetimeIndex.")

    if history.empty:
        raise PlotError("history is empty.")

    if not isinstance(forecast_df, pd.DataFrame):
        raise PlotError("forecast_df must be a pandas DataFrame.")

    if forecast_df.empty:
        raise PlotError("forecast_df is empty.")

    required_columns = {
        "date",
        "method",
        "horizon",
        "point",
        "lo80",
        "hi80",
        "lo95",
        "hi95",
    }
    missing_columns = required_columns - set(forecast_df.columns)

    if missing_columns:
        raise PlotError(
            f"forecast_df is missing required columns: {sorted(missing_columns)}"
        )

    historical_series = history.sort_index().copy()

    # Optionally show only the most recent section of history to keep the chart readable.
    if history_tail is not None:
        if history_tail <= 0:
            raise PlotError("history_tail must be a positive integer or None.")

        historical_series = historical_series.iloc[-history_tail:]

    forecast_output = forecast_df.copy()
    forecast_output["date"] = pd.to_datetime(forecast_output["date"])
    forecast_output = forecast_output.sort_values("date").reset_index(drop=True)

    # This plot is only intended for one method at a time.
    method_values = forecast_output["method"].dropna().astype(str).unique()
    if len(method_values) > 1:
        raise PlotError(
            "forecast_df should contain one method only. "
            f"Found methods: {sorted(method_values.tolist())}"
        )

    inferred_method_name = method_values[0] if len(method_values) == 1 else "forecast"

    # Convert internal method keys into cleaner display labels for chart legends.
    method_label_map = {
        "ets": "ETS",
        "seasonal_naive": "Seasonal naïve",
        "arima": "ARIMA",
    }

    if method_label is not None and str(method_label).strip():
        display_method_name = str(method_label).strip()
    else:
        display_method_name = method_label_map.get(
            inferred_method_name,
            inferred_method_name,
        )

    numeric_columns = ["point", "lo80", "hi80", "lo95", "hi95"]
    for column_name in numeric_columns:
        forecast_output[column_name] = pd.to_numeric(
            forecast_output[column_name],
            errors="coerce",
        )

    if forecast_output[numeric_columns].isna().any().any():
        raise PlotError("forecast_df contains non-numeric or missing forecast values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))

    # Plot the observed historical series.
    axis.plot(
        historical_series.index,
        historical_series.values,
        linewidth=2,
        label="History",
    )

    # Plot the wider 95% interval first so the narrower 80% interval sits on top.
    axis.fill_between(
        forecast_output["date"],
        forecast_output["lo95"],
        forecast_output["hi95"],
        alpha=0.15,
        label="95% interval",
    )
    axis.fill_between(
        forecast_output["date"],
        forecast_output["lo80"],
        forecast_output["hi80"],
        alpha=0.30,
        label="80% interval",
    )

    # Plot the central point forecast line.
    axis.plot(
        forecast_output["date"],
        forecast_output["point"],
        linewidth=2,
        label=f"Forecast ({display_method_name})",
    )

    axis.set_title(title)
    axis.set_xlabel("Month")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=True)

    # Build a footer containing source/provenance information for the export.
    footer_parts: list[str] = []

    if meta:
        cdid = meta.get("CDID")
        if cdid:
            footer_parts.append(f"CDID: {cdid}")

    footer_parts.append("Source: ONS time series download")
    footer_parts.append("Open Government Licence v3.0")
    footer_parts.append("Research use only; not financial or policy advice")

    figure.text(
        0.01,
        0.01,
        " | ".join(footer_parts),
        ha="left",
        va="bottom",
        fontsize=8,
        wrap=True,
    )

    figure.tight_layout(rect=[0, 0.06, 1, 1])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_path