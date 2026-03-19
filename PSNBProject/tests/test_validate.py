#python -m pytest PSNBProject/tests/test_validate.py -q -> run this on terminal to test
import pandas as pd
import pytest

from PSNBProject.validate import validate_monthly_series, ValidationError


def make_monthly_series(start="2024-01-01", periods=6, values=None):
    idx = pd.date_range(start=start, periods=periods, freq="MS")
    if values is None:
        values = list(range(periods))
    return pd.Series(values, index=idx, dtype="float64")


def test_validate_clean_monthly_series_passes():
    s = make_monthly_series(periods=6)

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.start == "2024-01-01"
    assert report.end == "2024-06-01"
    assert report.n_obs == 6
    assert report.n_missing_values == 0
    assert report.n_duplicate_timestamps == 0
    assert report.is_monotonic_increasing is True
    assert report.inferred_freq == "MS"
    assert report.expected_anchor == "MS"
    assert report.n_missing_months == 0
    assert report.missing_months == []
    assert report.warnings == []


def test_validate_detects_missing_month():
    s = make_monthly_series(periods=6).drop(pd.Timestamp("2024-03-01"))

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.start == "2024-01-01"
    assert report.end == "2024-06-01"
    assert report.n_obs == 5
    assert report.n_missing_months == 1
    assert report.missing_months == ["2024-03-01"]
    assert any("missing months" in w.lower() for w in report.warnings)


def test_validate_detects_nan_values():
    s = make_monthly_series(periods=6)
    s.iloc[2] = float("nan")

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.n_missing_values == 1
    assert any("missing values" in w.lower() or "nans" in w.lower() for w in report.warnings)


def test_validate_raises_on_duplicate_timestamps_strict():
    idx = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01"])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    with pytest.raises(ValidationError, match="duplicate timestamps"):
        validate_monthly_series(s, strict=True)


def test_validate_warns_on_duplicate_timestamps_non_strict():
    idx = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01"])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    report = validate_monthly_series(s, strict=False)

    assert report.n_duplicate_timestamps == 1
    assert any("duplicate timestamps" in w.lower() for w in report.warnings)


def test_validate_raises_on_unsorted_index_strict():
    s = make_monthly_series(periods=4).iloc[[2, 0, 1, 3]]

    with pytest.raises(ValidationError, match="not monotonic increasing"):
        validate_monthly_series(s, strict=True)


def test_validate_warns_on_unsorted_index_non_strict():
    s = make_monthly_series(periods=4).iloc[[2, 0, 1, 3]]

    report = validate_monthly_series(s, strict=False)

    assert report.is_monotonic_increasing is False
    assert any("not monotonic increasing" in w.lower() for w in report.warnings)


def test_validate_raises_on_non_datetime_index():
    s = pd.Series([1.0, 2.0, 3.0], index=["2024-01", "2024-02", "2024-03"])

    with pytest.raises(ValidationError, match="DatetimeIndex"):
        validate_monthly_series(s)


def test_validate_raises_on_non_series_input():
    with pytest.raises(ValidationError, match="pandas Series"):
        validate_monthly_series([1, 2, 3])  # type: ignore[arg-type]


def test_validate_warns_on_wrong_anchor_for_month_end_data():
    idx = pd.date_range(start="2024-01-31", periods=4, freq="ME")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.n_missing_months == 0
    assert any("month-start" in w.lower() for w in report.warnings)


def test_validate_month_end_series_passes_with_month_end_anchor():
    idx = pd.date_range(start="2024-01-31", periods=4, freq="ME")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    report = validate_monthly_series(s, expected_anchor="ME", strict=True)

    assert report.expected_anchor == "ME"
    assert report.n_missing_months == 0
    assert report.n_missing_values == 0
    assert report.n_duplicate_timestamps == 0
    assert report.warnings == []


def test_validate_accepts_expected_anchor_m_alias_for_month_end():
    idx = pd.date_range(start="2024-01-31", periods=4, freq="ME")
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    report = validate_monthly_series(s, expected_anchor="M", strict=True)

    assert report.expected_anchor == "M"
    assert report.warnings == []


def test_validate_raises_on_invalid_expected_anchor():
    s = make_monthly_series()

    with pytest.raises(ValidationError, match="expected_anchor"):
        validate_monthly_series(s, expected_anchor="W")


def test_validate_single_observation_series():
    s = pd.Series([100.0], index=[pd.Timestamp("2024-01-01")])

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.start == "2024-01-01"
    assert report.end == "2024-01-01"
    assert report.n_obs == 1
    assert report.n_missing_months == 0
    assert report.n_duplicate_timestamps == 0
    assert report.n_missing_values == 0
    assert report.warnings == []


def test_validate_empty_series():
    s = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    report = validate_monthly_series(s, expected_anchor="MS", strict=True)

    assert report.n_obs == 0
    assert report.n_missing_values == 0
    assert report.n_duplicate_timestamps == 0
    assert report.is_monotonic_increasing is True
    assert report.inferred_freq is None
    assert report.n_missing_months == 0
    assert report.missing_months == []
    assert report.warnings == []
    assert report.start == "NaT"
    assert report.end == "NaT"