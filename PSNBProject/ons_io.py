from __future__ import annotations

# ons_io.py
#
# This module loads an ONS time series download from CSV or Excel and converts
# it into a pandas Series indexed by monthly dates.
#
# The current implementation is designed for the standard two-column ONS
# download format where:
# - column 1 contains metadata labels and monthly period labels
# - column 2 contains metadata values or numeric observations


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import re

import pandas as pd


# Raised when an input file cannot be parsed into a usable monthly series.
class IOErrorPSNB(Exception):
    pass


# Stores the loaded monthly series along with source information and metadata.
@dataclass(frozen=True)
class LoadResult:
    series: pd.Series
    source_path: str
    meta: Dict[str, str]


# Match ONS monthly labels such as "2025 DEC".
MONTH_LABEL_PATTERN = re.compile(
    r"^(?P<year>\d{4})\s(?P<month_abbrev>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$"
)

# Convert ONS three-letter month abbreviations into month numbers.
MONTH_ABBREVIATION_TO_NUMBER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


# Read a CSV or Excel file into a pandas DataFrame.
def _read_tabular(
    path: Path,
    sheet_name: Optional[Union[str, int]] = None,
) -> pd.DataFrame:
    file_suffix = path.suffix.lower()

    if file_suffix == ".csv":
        return pd.read_csv(path)

    if file_suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)

    raise IOErrorPSNB(f"Unsupported file type: {file_suffix}. Use CSV or XLSX.")


# Detect whether the DataFrame looks like the standard two-column ONS download format.
def _looks_like_ons_timeseries_download(df: pd.DataFrame) -> bool:
    # The current loader expects exactly two columns.
    if df.shape[1] != 2:
        return False

    first_column_name = str(df.columns[0]).strip().lower()
    if first_column_name != "title":
        return False

    # The top rows in ONS downloads usually contain metadata keys such as CDID
    # and release information.
    first_column_values = {
        str(value).strip().upper()
        for value in df.iloc[:10, 0].tolist()
    }

    return (
        "CDID" in first_column_values
        or "SOURCE DATASET ID" in first_column_values
        or "RELEASE DATE" in first_column_values
    )


# Convert an ONS monthly label such as "2025 DEC" into a pandas Timestamp.
def _month_label_to_timestamp(month_label: str) -> pd.Timestamp:
    match = MONTH_LABEL_PATTERN.match(month_label)
    if match is None:
        raise IOErrorPSNB(f"Invalid ONS month label: {month_label}")

    year = int(match.group("year"))
    month_number = MONTH_ABBREVIATION_TO_NUMBER[match.group("month_abbrev")]

    # Use month-start timestamps by default.
    return pd.Timestamp(year=year, month=month_number, day=1)


# Parse the standard ONS two-column monthly time series format.
def _parse_ons_monthly_series(
    df: pd.DataFrame,
    *,
    anchor: str = "MS",
    dropna: bool = True,
) -> Tuple[pd.Series, Dict[str, str]]:
    label_column = df.columns[0]
    value_column = df.columns[1]

    # Extract selected metadata from the header block near the top of the file.
    metadata: Dict[str, str] = {}
    metadata_keys = {
        "CDID",
        "SOURCE DATASET ID",
        "PREUNIT",
        "UNIT",
        "RELEASE DATE",
        "NEXT RELEASE",
        "IMPORTANT NOTES",
        "TITLE",
    }

    for row_index in range(min(50, len(df))):
        raw_key = str(df.at[row_index, label_column]).strip()
        raw_value = df.at[row_index, value_column]

        if raw_key and raw_key.upper() in metadata_keys:
            metadata[raw_key] = "" if pd.isna(raw_value) else str(raw_value).strip()

    # Keep only rows that match the ONS monthly label pattern.
    label_series = df[label_column].astype(str).str.strip().str.upper()
    is_month_row = label_series.str.match(MONTH_LABEL_PATTERN)

    if not is_month_row.any():
        raise IOErrorPSNB("Could not find any monthly rows like 'YYYY MON' in this file.")

    monthly_data = df.loc[is_month_row, [label_column, value_column]].copy()
    monthly_data[label_column] = monthly_data[label_column].astype(str).str.strip().str.upper()

    # Convert the observation column to numeric values.
    monthly_data[value_column] = pd.to_numeric(monthly_data[value_column], errors="coerce")

    # Convert ONS month labels into timestamps.
    monthly_data["date"] = monthly_data[label_column].map(_month_label_to_timestamp)

    # Allow the caller to choose month-start or month-end index anchoring.
    if anchor is not None:
        anchor = anchor.upper()

        if anchor == "MS":
            pass
        elif anchor in ("M", "ME"):
            monthly_data["date"] = (
                monthly_data["date"].dt.to_period("M").dt.to_timestamp("M")
            )
        else:
            raise IOErrorPSNB("anchor must be 'MS', 'M'/'ME', or None.")

    # Drop rows where the observation value could not be read as numeric.
    if dropna:
        monthly_data = monthly_data.dropna(subset=[value_column])

    # Sort by date and keep the most recent version if duplicates exist.
    monthly_data = (
        monthly_data.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

    monthly_series = pd.Series(
        monthly_data[value_column].to_numpy(),
        index=pd.DatetimeIndex(monthly_data["date"]),
        name=str(value_column),
    )
    monthly_series.index.name = "date"

    # Add a couple of helpful metadata items for later provenance/reporting use.
    metadata.setdefault("Series column", str(value_column))
    metadata.setdefault("Format", "ONS time series download")

    return monthly_series, metadata


# Load a monthly time series from CSV or Excel and return the parsed result.
def load_series(
    file_path: Union[str, Path],
    *,
    sheet_name: Optional[Union[str, int]] = None,
    anchor: str = "MS",
    dropna: bool = True,
) -> LoadResult:
    path = Path(file_path)

    if not path.exists():
        raise IOErrorPSNB(f"File not found: {path}")

    input_df = _read_tabular(path, sheet_name=sheet_name)

    if input_df.empty:
        raise IOErrorPSNB("Input file is empty.")

    # At the moment, the loader only supports the recognised ONS two-column format.
    if _looks_like_ons_timeseries_download(input_df):
        monthly_series, metadata = _parse_ons_monthly_series(
            input_df,
            anchor=anchor,
            dropna=dropna,
        )
        return LoadResult(series=monthly_series, source_path=str(path), meta=metadata)

    # Generic column-based loading could be added later if needed.
    raise IOErrorPSNB(
        "This file does not look like an ONS time series download. "
        "For generic CSV/XLSX, add a date/value column loader."
    )