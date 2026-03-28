from __future__ import annotations

# main.py
#
# This is the command-line entry point for the project.
#
# It runs the full batch pipeline:
# 1. load the ONS series
# 2. validate the monthly structure
# 3. generate final forecasts for each active method
# 4. run expanding-window backtests
# 5. compute error and coverage metrics
# 6. export CSV, JSON, and PNG outputs into a timestamped run folder

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import expanding_window_backtest
from metrics import (
    compute_coverage,
    compute_coverage_summary,
    compute_error_summary,
    compute_errors,
)
from models import forecast_ets, forecast_seasonal_naive
# from models import forecast_arima  # retained for explanation; not active in final scope
from ons_io import load_series
from plot import plot_forecast
from validate import validate_monthly_series


# Define core project paths relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "series-050326.csv"

# Define the default forecast and backtest settings used by the batch pipeline.
FORECAST_H = 12
BACKTEST_H = 12
MIN_TRAIN = 60

# Keep ARIMA visible but inactive so the final submitted scope stays stable.
METHODS = {
    "seasonal_naive": forecast_seasonal_naive,
    "ets": forecast_ets,
    # "arima": forecast_arima,
}

# Use cleaner labels for exported charts and human-readable output.
METHOD_LABELS = {
    "seasonal_naive": "Seasonal naïve",
    "ets": "ETS",
    # "arima": "ARIMA",
}


# Create a timestamped run folder inside the outputs directory.
def build_run_dir(outputs_root: Path) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = outputs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


# Save a dictionary to JSON with indentation for readability.
def export_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    # Create a fresh output folder for this run before generating any files.
    run_dir = build_run_dir(OUTPUTS_ROOT)

    # Record the run configuration so the batch output is reproducible later.
    config = {
        "data_path": str(DATA_PATH),
        "forecast_h": FORECAST_H,
        "backtest_h": BACKTEST_H,
        "min_train": MIN_TRAIN,
        "methods": list(METHODS.keys()),
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    export_json(run_dir / "config.json", config)

    # Load the input time series from the project dataset file.
    load_result = load_series(DATA_PATH)
    full_series = load_result.series.sort_index()

    # Validate the monthly series structure before any modelling is attempted.
    validation_report = validate_monthly_series(
        full_series,
        expected_anchor="MS",
        strict=True,
    )

    # Generate one final forecast for each active method using the full series.
    forecast_frames = [
        forecast_function(full_series, h=FORECAST_H)
        for forecast_function in METHODS.values()
    ]
    forecast_df = pd.concat(forecast_frames, ignore_index=True)
    forecast_df = forecast_df.sort_values(["method", "date"]).reset_index(drop=True)
    forecast_df.to_csv(run_dir / "forecast.csv", index=False)

    # Run the expanding-window backtest to compare forecasts with realised outcomes.
    prediction_df = expanding_window_backtest(
        full_series,
        h_max=BACKTEST_H,
        methods=METHODS,
        min_train=MIN_TRAIN,
    )
    prediction_df = prediction_df.sort_values(
        ["method", "origin", "horizon"]
    ).reset_index(drop=True)
    prediction_df.to_csv(run_dir / "backtest_predictions.csv", index=False)

    # Compute point forecast error metrics and interval performance metrics.
    errors_df = compute_errors(prediction_df)
    coverage_df = compute_coverage(prediction_df)
    error_summary_df = compute_error_summary(prediction_df)
    coverage_summary_df = compute_coverage_summary(prediction_df)

    errors_df.to_csv(run_dir / "errors.csv", index=False)
    coverage_df.to_csv(run_dir / "coverage.csv", index=False)
    error_summary_df.to_csv(run_dir / "error_summary.csv", index=False)
    coverage_summary_df.to_csv(run_dir / "coverage_summary.csv", index=False)

    # Export the validation report so dataset quality checks are recorded.
    validation_report.to_json(str(run_dir / "validation_report.json"))

    # Create one exported PNG forecast chart per active method.
    for method_name in forecast_df["method"].dropna().unique():
        method_forecast_df = forecast_df.loc[forecast_df["method"] == method_name].copy()
        method_label = METHOD_LABELS.get(method_name, method_name)

        plot_forecast(
            history=full_series,
            forecast_df=method_forecast_df,
            output_path=run_dir / f"forecast_{method_name}.png",
            title=f"PSNB Forecast ({method_label})",
            ylabel="PSNB",
            meta=load_result.meta,
            method_label=method_label,
        )

    # Summarise the run outputs in a compact JSON file for traceability.
    run_summary = {
        "input_path": str(load_result.source_path),
        "run_dir": str(run_dir),
        "series_start": str(full_series.index.min().date()),
        "series_end": str(full_series.index.max().date()),
        "series_length": int(len(full_series)),
        "forecast_rows": int(len(forecast_df)),
        "backtest_rows": int(len(prediction_df)),
        "error_rows": int(len(errors_df)),
        "coverage_rows": int(len(coverage_df)),
        "error_summary_rows": int(len(error_summary_df)),
        "coverage_summary_rows": int(len(coverage_summary_df)),
        "validation_warning_count": int(len(validation_report.warnings)),
        "methods_run": list(METHODS.keys()),
    }
    export_json(run_dir / "run_summary.json", run_summary)

    # Print a short terminal summary so the user can confirm the run succeeded.
    print("Run complete.")
    print(f"Input file: {load_result.source_path}")
    print(f"Run folder: {run_dir}")
    print(
        f"Series span: {full_series.index.min().strftime('%Y-%m')} "
        f"to {full_series.index.max().strftime('%Y-%m')}"
    )
    print(f"Methods: {', '.join(METHODS.keys())}")
    print(f"Forecast rows: {len(forecast_df)}")
    print(f"Backtest rows: {len(prediction_df)}")
    print(f"Error summary rows: {len(error_summary_df)}")
    print(f"Coverage summary rows: {len(coverage_summary_df)}")
    print(f"Validation warnings: {len(validation_report.warnings)}")

    # Print validation warnings in detail if any were found.
    if validation_report.warnings:
        print("Warning details:")
        for warning in validation_report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()