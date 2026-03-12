from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import re

import pandas as pd


class IOErrorPSNB(Exception):
    """Raised when input files cannot be parsed into a usable monthly series."""


@dataclass(frozen=True)
class LoadResult:
    series: pd.Series
    source_path: str
    meta: Dict[str, str]


# Matches ONS monthly period labels like "2025 DEC"
_MONTH_RE = re.compile(r"^(?P<year>\d{4})\s(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$")
_MON_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}


def _read_tabular(path: Path, sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    raise IOErrorPSNB(f"Unsupported file type: {suffix}. Use CSV or XLSX.")


def _looks_like_ons_timeseries_download(df: pd.DataFrame) -> bool:
    # Typical ONS format: first column is "Title" and contains keys like CDID
    if df.shape[1] != 2:
        return False
    c0 = str(df.columns[0]).strip().lower()
    if c0 != "title":
        return False
    first_vals = set(str(x).strip().upper() for x in df.iloc[:10, 0].tolist())
    return "CDID" in first_vals or "SOURCE DATASET ID" in first_vals or "RELEASE DATE" in first_vals


def _parse_ons_monthly_series(df: pd.DataFrame, *, anchor: str = "MS", dropna: bool = True) -> Tuple[pd.Series, Dict[str, str]]:
    label_col = df.columns[0]
    value_col = df.columns[1]

    # Extract metadata from the header block (key in col0, value in col1)
    meta: Dict[str, str] = {}
    for i in range(min(50, len(df))):
        k = str(df.at[i, label_col]).strip()
        v = df.at[i, value_col]
        if k and k.upper() in {"CDID", "SOURCE DATASET ID", "PREUNIT", "UNIT", "RELEASE DATE", "NEXT RELEASE", "IMPORTANT NOTES", "TITLE"}:
            meta[k] = "" if pd.isna(v) else str(v).strip()

    # Filter monthly rows only
    labels = df[label_col].astype(str).str.strip().str.upper()
    is_month = labels.str.match(_MONTH_RE)

    if not is_month.any():
        raise IOErrorPSNB("Could not find any monthly rows like 'YYYY MON' in this file.")

    month_df = df.loc[is_month, [label_col, value_col]].copy()
    month_df[label_col] = month_df[label_col].astype(str).str.strip().str.upper()
    month_df[value_col] = pd.to_numeric(month_df[value_col], errors="coerce")

    # Convert "YYYY MON" -> Timestamp
    def to_ts(s: str) -> pd.Timestamp:
        m = _MONTH_RE.match(s)
        assert m is not None
        y = int(m.group("year"))
        mon = _MON_MAP[m.group("mon")]
        # month start by default
        return pd.Timestamp(year=y, month=mon, day=1)

    month_df["date"] = month_df[label_col].map(to_ts)

    if anchor is not None:
        if anchor.upper() == "MS":
            # already MS
            pass
        elif anchor.upper() in ("M", "ME"):
            month_df["date"] = month_df["date"].dt.to_period("M").dt.to_timestamp("M")
        else:
            raise IOErrorPSNB("anchor must be 'MS', 'M'/'ME', or None.")

    if dropna:
        month_df = month_df.dropna(subset=[value_col])

    month_df = month_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    s = pd.Series(month_df[value_col].to_numpy(), index=pd.DatetimeIndex(month_df["date"]), name=str(value_col))
    s.index.name = "date"

    # Helpful extra metadata for provenance/use later
    meta.setdefault("Series column", str(value_col))
    meta.setdefault("Format", "ONS time series download")

    return s, meta


def load_series(
    file_path: Union[str, Path],
    *,
    sheet_name: Optional[Union[str, int]] = None,
    anchor: str = "MS",
    dropna: bool = True,
) -> LoadResult:
    """
    Load a monthly time series from CSV/XLSX.

    Automatically supports ONS time-series download format (like your DZLS file),
    and returns a monthly Series indexed by month start ('MS') by default.
    """
    path = Path(file_path)
    if not path.exists():
        raise IOErrorPSNB(f"File not found: {path}")

    df = _read_tabular(path, sheet_name=sheet_name)
    if df.empty:
        raise IOErrorPSNB("Input file is empty.")

    if _looks_like_ons_timeseries_download(df):
        s, meta = _parse_ons_monthly_series(df, anchor=anchor, dropna=dropna)
        return LoadResult(series=s, source_path=str(path), meta=meta)

    # If not ONS format, you can later add generic column-based loading here.
    raise IOErrorPSNB(
        "This file does not look like an ONS time series download. "
        "For generic CSV/XLSX, add a date/value column loader."
    ) 