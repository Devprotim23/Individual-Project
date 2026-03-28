from __future__ import annotations

# validate.py
#
# This module validates monthly time series before they are passed into the
# forecasting and backtesting pipeline.
#
# The goal is to catch structural issues early, such as:
# - non-datetime indexes
# - duplicate timestamps
# - unsorted dates
# - missing values
# - missing months
# - incorrect monthly anchoring

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import json

import pandas as pd


# Raised when a series fails a validation rule required for modelling.
class ValidationError(Exception):
    pass


# Stores a structured summary of validation results.
@dataclass(frozen=True)
class ValidationReport:
    start: str
    end: str
    n_obs: int
    n_missing_values: int
    n_duplicate_timestamps: int
    is_monotonic_increasing: bool
    inferred_freq: Optional[str]
    expected_anchor: str
    n_missing_months: int
    missing_months: List[str]
    warnings: List[str]

    # Convert the validation report into a plain Python dictionary.
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Save the validation report as a JSON file.
    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# Build an inclusive month-start date range from the first month to the last month.
def _month_range_ms(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # Normalise both dates to the first day of their month.
    start_month_start = pd.Timestamp(start.year, start.month, 1)
    end_month_start = pd.Timestamp(end.year, end.month, 1)

    return pd.date_range(start=start_month_start, end=end_month_start, freq="MS")


# Validate that a pandas Series has a suitable monthly structure for forecasting.
def validate_monthly_series(
    series: pd.Series,
    *,
    expected_anchor: str = "MS",
    strict: bool = True,
) -> ValidationReport:
    warnings: List[str] = []

    if not isinstance(series, pd.Series):
        raise ValidationError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValidationError("Series index must be a pandas DatetimeIndex.")

    expected_anchor = expected_anchor.upper()
    if expected_anchor not in ("MS", "M", "ME"):
        raise ValidationError("expected_anchor must be 'MS' or 'M'/'ME'.")

    # Count duplicate timestamps because duplicates can make forecasting ambiguous.
    n_duplicate_timestamps = int(series.index.duplicated().sum())
    if n_duplicate_timestamps > 0:
        message = f"Series index contains {n_duplicate_timestamps} duplicate timestamps."
        if strict:
            raise ValidationError(message)
        warnings.append(message)

    # Check that dates are already ordered from earliest to latest.
    is_monotonic_increasing = bool(series.index.is_monotonic_increasing)
    if not is_monotonic_increasing:
        message = "Series index is not monotonic increasing (not sorted by date)."
        if strict:
            raise ValidationError(message)
        warnings.append(message)

    # Work with a sorted copy so later checks behave consistently.
    sorted_series = series.sort_index()
    start = sorted_series.index.min()
    end = sorted_series.index.max()

    # Handle an empty series safely without trying to infer dates or frequency.
    if len(sorted_series) == 0:
        return ValidationReport(
            start="NaT",
            end="NaT",
            n_obs=0,
            n_missing_values=0,
            n_duplicate_timestamps=n_duplicate_timestamps,
            is_monotonic_increasing=is_monotonic_increasing,
            inferred_freq=None,
            expected_anchor=expected_anchor,
            n_missing_months=0,
            missing_months=[],
            warnings=warnings,
        )

    # Count missing values because forecasting methods usually expect complete input.
    n_missing_values = int(sorted_series.isna().sum())
    if n_missing_values > 0:
        warnings.append(f"Series contains {n_missing_values} missing values (NaNs).")

    # Try to infer the time frequency from the index.
    inferred_freq = None
    try:
        inferred_freq = pd.infer_freq(sorted_series.index)
    except Exception:
        inferred_freq = None

    # Check whether the timestamps appear to be anchored correctly.
    # For month-start series, dates should usually fall on day 1.
    # For month-end series, dates should usually match the final day of each month.
    sample_index = sorted_series.index[: min(24, len(sorted_series))]

    if expected_anchor == "MS":
        if not all(timestamp.day == 1 for timestamp in sample_index):
            warnings.append(
                "Index does not appear to be anchored to month-start (day != 1)."
            )
    else:
        if not all(
            timestamp == timestamp.to_period("M").to_timestamp("M")
            for timestamp in sample_index
        ):
            warnings.append("Index does not appear to be anchored to month-end.")

    # Build the full monthly range that should exist between the start and end dates.
    full_month_range = _month_range_ms(start, end)

    # Convert observed timestamps to month-start equivalents so missing months can
    # be detected consistently even if the original series used month-end dates.
    observed_month_starts = sorted_series.index.to_period("M").to_timestamp(how="start")
    observed_unique_month_starts = pd.DatetimeIndex(
        observed_month_starts.unique()
    ).sort_values()

    missing_months = full_month_range.difference(observed_unique_month_starts)
    n_missing_months = int(len(missing_months))

    if n_missing_months > 0:
        warnings.append(f"Series has {n_missing_months} missing months in the date range.")

    return ValidationReport(
        start=str(start.date()),
        end=str(end.date()),
        n_obs=int(len(sorted_series)),
        n_missing_values=n_missing_values,
        n_duplicate_timestamps=n_duplicate_timestamps,
        is_monotonic_increasing=is_monotonic_increasing,
        inferred_freq=inferred_freq,
        expected_anchor=expected_anchor,
        n_missing_months=n_missing_months,
        missing_months=[timestamp.strftime("%Y-%m-%d") for timestamp in missing_months],
        warnings=warnings,
    )