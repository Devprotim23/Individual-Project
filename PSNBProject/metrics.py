from __future__ import annotations

import numpy as np
import pandas as pd


class MetricsError(Exception):
    """Raised when metric computation cannot be performed safely."""


def compute_errors(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute point-forecast error metrics by method and horizon.

    Returns
    -------
    pd.DataFrame
        Columns:
        method, horizon, n, mae, rmse
    """
    required = {"method", "horizon", "point", "actual"}
    missing = required - set(pred_df.columns)
    if missing:
        raise MetricsError(f"pred_df is missing required columns: {sorted(missing)}")

    df = pred_df.copy()
    df = df.dropna(subset=["point", "actual"])

    if df.empty:
        raise MetricsError("No valid rows available to compute errors.")

    df["abs_error"] = (df["actual"] - df["point"]).abs()
    df["sq_error"] = (df["actual"] - df["point"]) ** 2

    out = (
        df.groupby(["method", "horizon"], as_index=False)
        .agg(
            n=("actual", "size"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
        )
    )

    out["rmse"] = np.sqrt(out["mse"])
    out = out.drop(columns=["mse"])

    return out.sort_values(["method", "horizon"]).reset_index(drop=True)


def compute_coverage(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interval coverage and average interval width by method and horizon.

    Returns
    -------
    pd.DataFrame
        Columns:
        method, horizon, n,
        coverage80, avg_width80,
        coverage95, avg_width95
    """
    required = {"method", "horizon", "actual", "lo80", "hi80", "lo95", "hi95"}
    missing = required - set(pred_df.columns)
    if missing:
        raise MetricsError(f"pred_df is missing required columns: {sorted(missing)}")

    df = pred_df.copy()
    df = df.dropna(subset=["actual", "lo80", "hi80", "lo95", "hi95"])

    if df.empty:
        raise MetricsError("No valid rows available to compute coverage.")

    df["in80"] = ((df["actual"] >= df["lo80"]) & (df["actual"] <= df["hi80"])).astype(float)
    df["in95"] = ((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).astype(float)
    df["width80"] = df["hi80"] - df["lo80"]
    df["width95"] = df["hi95"] - df["lo95"]

    out = (
        df.groupby(["method", "horizon"], as_index=False)
        .agg(
            n=("actual", "size"),
            coverage80=("in80", "mean"),
            avg_width80=("width80", "mean"),
            coverage95=("in95", "mean"),
            avg_width95=("width95", "mean"),
        )
    )

    return out.sort_values(["method", "horizon"]).reset_index(drop=True)