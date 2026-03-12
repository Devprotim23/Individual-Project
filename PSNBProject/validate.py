from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json

import pandas as pd


class ValidationError(Exception):
    """Raised when a series fails a validation rule required for modelling."""


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _month_range_ms(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Inclusive month-start date range.

    Uses a robust approach compatible across pandas versions:
    - month start is always YYYY-MM-01
    """
    start_ms = pd.Timestamp(start.year, start.month, 1)
    end_ms = pd.Timestamp(end.year, end.month, 1)
    return pd.date_range(start=start_ms, end=end_ms, freq="MS")


def validate_monthly_series(
    s: pd.Series,
    *,
    expected_anchor: str = "MS",
    strict: bool = True,
) -> ValidationReport:
    """
    Validate a monthly time series suitable for forecasting/backtesting.

    Parameters
    ----------
    s:
        pandas Series with DatetimeIndex.
    expected_anchor:
        'MS' for month-start or 'M'/'ME' for month-end.
    strict:
        If True, raise ValidationError on critical issues (non-datetime index, unsorted, duplicates).
        If False, return report with warnings where possible.

    Returns
    -------
    ValidationReport
    """
    warnings: List[str] = []

    if not isinstance(s, pd.Series):
        raise ValidationError("Input must be a pandas Series.")

    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValidationError("Series index must be a pandas DatetimeIndex.")

    # duplicates
    n_dupes = int(s.index.duplicated().sum())
    if n_dupes > 0:
        msg = f"Series index contains {n_dupes} duplicate timestamps."
        if strict:
            raise ValidationError(msg)
        warnings.append(msg)

    # sortedness
    is_sorted = bool(s.index.is_monotonic_increasing)
    if not is_sorted:
        msg = "Series index is not monotonic increasing (not sorted by date)."
        if strict:
            raise ValidationError(msg)
        warnings.append(msg)

    s_sorted = s.sort_index()
    start = s_sorted.index.min()
    end = s_sorted.index.max()

    # missing values (NaNs)
    n_missing_values = int(s_sorted.isna().sum())
    if n_missing_values > 0:
        warnings.append(f"Series contains {n_missing_values} missing values (NaNs).")

    # inferred frequency (may be None)
    inferred = None
    try:
        inferred = pd.infer_freq(s_sorted.index)
    except Exception:
        inferred = None

    expected_anchor = expected_anchor.upper()
    if expected_anchor not in ("MS", "M", "ME"):
        raise ValidationError("expected_anchor must be 'MS' or 'M'/'ME'.")

    # anchor check (soft warning)
    sample = s_sorted.index[: min(24, len(s_sorted))]
    if expected_anchor == "MS":
        if not all(d.day == 1 for d in sample):
            warnings.append("Index does not appear to be anchored to month-start (day != 1).")
    else:
        # month-end check
        if not all(d == d.to_period("M").to_timestamp("M") for d in sample):
            warnings.append("Index does not appear to be anchored to month-end.")

    # missing months check (based on month-start)
    full = _month_range_ms(start, end)

    # Convert observed timestamps to month-start timestamps
    have = s_sorted.index.to_period("M").to_timestamp(how="start")
    have_unique = pd.DatetimeIndex(have.unique()).sort_values()

    missing = full.difference(have_unique)
    n_missing_months = int(len(missing))
    if n_missing_months > 0:
        warnings.append(f"Series has {n_missing_months} missing months in the date range.")

    return ValidationReport(
        start=str(start.date()),
        end=str(end.date()),
        n_obs=int(len(s_sorted)),
        n_missing_values=n_missing_values,
        n_duplicate_timestamps=n_dupes,
        is_monotonic_increasing=is_sorted,
        inferred_freq=inferred,
        expected_anchor=expected_anchor,
        n_missing_months=n_missing_months,
        missing_months=[d.strftime("%Y-%m-%d") for d in missing],
        warnings=warnings,
    )