from __future__ import annotations

# app.py is the Streamlit user interface for the project.
# It supports forecast generation and historical evaluation workflows.

import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ons_io import load_series
from validate import validate_monthly_series
from models import forecast_seasonal_naive, forecast_ets
# from models import forecast_arima  # retained for explanation; not active in final scope
from plot import plot_forecast


# Configure the Streamlit page before any UI elements are rendered.
st.set_page_config(
    page_title="PSNB Forecasting Tool",
    layout="wide",
)


# Define the default dataset path relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "series-050326.csv"

# Keep ARIMA visible but inactive so the final submitted scope stays stable.
METHODS = {
    "seasonal_naive": forecast_seasonal_naive,
    "ets": forecast_ets,
    # "arima": forecast_arima,
}

# Use cleaner labels in the GUI and exported outputs.
METHOD_LABELS = {
    "seasonal_naive": "Seasonal naïve",
    "ets": "ETS",
    # "arima": "ARIMA",
}


# Load and validate the default dataset once, then cache the result for reuse.
@st.cache_data(show_spinner=False)
def load_default_series():
    load_result = load_series(DEFAULT_DATA_PATH)
    validation_report = validate_monthly_series(
        load_result.series,
        expected_anchor="MS",
        strict=True,
    )
    return load_result, validation_report


# Build the month choices shown in the origin dropdowns.
def month_options(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return list(index.sort_values().unique())


# Format a timestamp as YYYY-MM for display in the GUI.
def month_label(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m")


# Render the Matplotlib forecast chart to a temporary PNG so Streamlit can display
# and download it without saving a permanent intermediate file.
def render_plot_image(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    title: str,
    history_tail: int | None,
    meta: dict,
    method_label: str | None = None,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        plot_forecast(
            history=history,
            forecast_df=forecast_df,
            output_path=temp_path,
            title=title,
            ylabel="PSNB",
            history_tail=history_tail,
            meta=meta,
            method_label=method_label,
        )
        image_bytes = temp_path.read_bytes()
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return image_bytes


# Evaluate one historical forecast run against the realised values that actually
# occurred in the original full series.
def evaluate_single_run(
    full_series: pd.Series,
    forecast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    evaluation_df = forecast_df.copy()
    evaluation_df["date"] = pd.to_datetime(evaluation_df["date"])
    evaluation_df["actual"] = full_series.reindex(evaluation_df["date"]).values

    # Compute point forecast errors for the realised forecast window.
    evaluation_df["abs_error"] = (evaluation_df["actual"] - evaluation_df["point"]).abs()
    evaluation_df["sq_error"] = (evaluation_df["actual"] - evaluation_df["point"]) ** 2

    # Record whether each realised value fell inside the forecast intervals.
    evaluation_df["hit80"] = (
        (evaluation_df["actual"] >= evaluation_df["lo80"])
        & (evaluation_df["actual"] <= evaluation_df["hi80"])
    ).astype(float)

    evaluation_df["hit95"] = (
        (evaluation_df["actual"] >= evaluation_df["lo95"])
        & (evaluation_df["actual"] <= evaluation_df["hi95"])
    ).astype(float)

    # Interval width helps show how much uncertainty the model produced.
    evaluation_df["width80"] = evaluation_df["hi80"] - evaluation_df["lo80"]
    evaluation_df["width95"] = evaluation_df["hi95"] - evaluation_df["lo95"]

    valid_rows = evaluation_df.dropna(subset=["actual"]).copy()

    metrics = {
        "mae": float(valid_rows["abs_error"].mean()) if not valid_rows.empty else np.nan,
        "rmse": float(np.sqrt(valid_rows["sq_error"].mean())) if not valid_rows.empty else np.nan,
        "coverage80": float(valid_rows["hit80"].mean()) if not valid_rows.empty else np.nan,
        "coverage95": float(valid_rows["hit95"].mean()) if not valid_rows.empty else np.nan,
        "avg_width80": float(valid_rows["width80"].mean()) if not valid_rows.empty else np.nan,
        "avg_width95": float(valid_rows["width95"].mean()) if not valid_rows.empty else np.nan,
        "n_actual": int(valid_rows["actual"].notna().sum()),
    }

    return evaluation_df, metrics


# Build the combined chart data for the lower historical evaluation chart.
# This includes recent pre-origin history, the forecast path, realised values,
# and the 80% interval for the forecast window.
def build_evaluation_plot_df(
    train_series: pd.Series,
    full_series: pd.Series,
    forecast_df: pd.DataFrame,
    history_tail: int | None = None,
) -> pd.DataFrame:
    forecast_dates = pd.to_datetime(forecast_df["date"])
    actual_series = full_series.reindex(forecast_dates)

    history_view = train_series.sort_index().copy()
    if history_tail is not None:
        history_view = history_view.iloc[-history_tail:]

    history_df = pd.DataFrame(
        {
            "date": history_view.index,
            "History": history_view.values,
            "Forecast": np.nan,
            "Actual": np.nan,
            "Lo80": np.nan,
            "Hi80": np.nan,
        }
    )

    forecast_window_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "History": np.nan,
            "Forecast": forecast_df["point"].values,
            "Actual": actual_series.values,
            "Lo80": forecast_df["lo80"].values,
            "Hi80": forecast_df["hi80"].values,
        }
    )

    combined_df = pd.concat([history_df, forecast_window_df], ignore_index=True)
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    return combined_df


# Build the data for the separate "What actually happened" chart.
# This chart only shows the realised values for the same future months as the
# historical forecast window.
def build_actual_window_plot_df(
    full_series: pd.Series,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    forecast_dates = pd.to_datetime(forecast_df["date"])
    actual_series = full_series.reindex(forecast_dates)

    return pd.DataFrame(
        {
            "date": forecast_dates,
            "Actual": actual_series.values,
        }
    ).dropna(subset=["Actual"])


# Build the comparison chart data for all active methods from the same
# historical origin and horizon.
def build_comparison_plot_df(
    train_series: pd.Series,
    full_series: pd.Series,
    forecast_results: dict[str, pd.DataFrame],
    history_tail: int | None = None,
) -> pd.DataFrame:
    history_view = train_series.sort_index().copy()
    if history_tail is not None:
        history_view = history_view.iloc[-history_tail:]

    frames: list[pd.DataFrame] = []

    history_df = pd.DataFrame(
        {
            "date": history_view.index,
            "Series": "History",
            "value": history_view.values,
        }
    )
    frames.append(history_df)

    # Use the first forecast output to align the realised values for the common horizon.
    first_method_key = next(iter(forecast_results))
    first_forecast_df = forecast_results[first_method_key].copy()
    first_forecast_df["date"] = pd.to_datetime(first_forecast_df["date"])
    actual_series = full_series.reindex(first_forecast_df["date"])

    actual_df = pd.DataFrame(
        {
            "date": first_forecast_df["date"],
            "Series": "Actual",
            "value": actual_series.values,
        }
    )
    frames.append(actual_df)

    for method_key, forecast_df in forecast_results.items():
        method_label = METHOD_LABELS.get(method_key, method_key)

        method_forecast_df = forecast_df.copy()
        method_forecast_df["date"] = pd.to_datetime(method_forecast_df["date"])

        model_line_df = pd.DataFrame(
            {
                "date": method_forecast_df["date"],
                "Series": method_label,
                "value": method_forecast_df["point"].values,
            }
        )
        frames.append(model_line_df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.sort_values(["date", "Series"]).reset_index(drop=True)
    return combined_df


# Build the comparison metrics table shown when all active methods are run together.
def build_comparison_metrics_table(
    metrics_by_method: dict[str, dict[str, float]]
) -> pd.DataFrame:
    rows = []

    for method_key, metrics in metrics_by_method.items():
        rows.append(
            {
                "Method": METHOD_LABELS.get(method_key, method_key),
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "Coverage 80%": metrics["coverage80"],
                "Coverage 95%": metrics["coverage95"],
                "Average width 80%": metrics["avg_width80"],
                "Average width 95%": metrics["avg_width95"],
                "Actuals available": metrics["n_actual"],
            }
        )

    output_df = pd.DataFrame(rows)

    if not output_df.empty:
        numeric_2dp = ["MAE", "RMSE", "Average width 80%", "Average width 95%"]
        numeric_3dp = ["Coverage 80%", "Coverage 95%"]

        for column_name in numeric_2dp:
            output_df[column_name] = pd.to_numeric(
                output_df[column_name],
                errors="coerce",
            ).round(2)

        for column_name in numeric_3dp:
            output_df[column_name] = pd.to_numeric(
                output_df[column_name],
                errors="coerce",
            ).round(3)

    return output_df


# Format date columns as YYYY-MM before displaying or exporting tables.
def format_table_dates(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    if "date" in output_df.columns:
        output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime("%Y-%m")

    if "origin" in output_df.columns:
        output_df["origin"] = pd.to_datetime(output_df["origin"]).dt.strftime("%Y-%m")

    return output_df


# Convert a DataFrame into UTF-8 encoded CSV bytes for Streamlit download buttons.
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.title("PSNB Forecasting and Historical Evaluation Tool")
    st.caption(
        "Generate monthly PSNB forecasts from ONS data and evaluate forecast performance "
        "against known historical outcomes."
    )

    # Load the default dataset once and stop early if the file cannot be used.
    try:
        load_result, validation_report = load_default_series()
    except Exception as exc:
        st.error(f"Failed to load default dataset: {exc}")
        st.stop()

    full_series = load_result.series.sort_index()

    # -------------------------
    # Sidebar control panel
    # -------------------------
    with st.sidebar:
        st.header("Controls")

        analysis_mode = st.radio(
            "Analysis mode",
            options=["Forecast generation", "Historical evaluation"],
            index=0,
        )

        compare_methods = False

        # Comparison mode is only relevant in historical evaluation because it
        # compares multiple models against known realised outcomes.
        if analysis_mode == "Historical evaluation":
            compare_methods = st.checkbox(
                "Compare all active methods",
                value=False,
                disabled=len(METHODS) < 2,
                help=(
                    "Runs the historical evaluation for every currently enabled "
                    "model from the same origin and horizon."
                ),
            )

            if len(METHODS) < 2:
                st.caption(
                    "Comparison mode becomes available when at least two forecast methods are enabled."
                )

        # Disable the single-method selector when comparison mode is active.
        selected_method_key = st.selectbox(
            "Forecast method",
            options=list(METHODS.keys()),
            index=1 if "ets" in METHODS else 0,
            format_func=lambda key: METHOD_LABELS.get(key, key),
            disabled=compare_methods,
        )

        forecast_horizon = st.slider(
            "Forecast horizon (months)",
            min_value=1,
            max_value=12,
            value=12,
        )

        history_window_label = st.selectbox(
            "History shown on chart",
            options=["36 months", "60 months", "All history"],
            index=1,
        )

        history_tail = {
            "36 months": 36,
            "60 months": 60,
            "All history": None,
        }[history_window_label]

        available_months = month_options(full_series.index)

        # In forecast generation mode, the user can choose any available origin month.
        if analysis_mode == "Forecast generation":
            default_origin_index = len(available_months) - 1
            selected_origin_label = st.selectbox(
                "Forecast origin month",
                options=[month_label(ts) for ts in available_months],
                index=default_origin_index,
            )

        # In historical evaluation mode, only show origin months where enough
        # future realised data exists to evaluate the chosen horizon.
        else:
            max_eval_index = max(0, len(available_months) - forecast_horizon - 1)
            default_origin_index = max_eval_index

            selected_origin_label = st.selectbox(
                "Evaluation origin month",
                options=[month_label(ts) for ts in available_months[: max_eval_index + 1]],
                index=default_origin_index,
            )

        run_button = st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
        )

    # -------------------------
    # Dataset status panel
    # -------------------------
    warning_count = len(getattr(validation_report, "warnings", []))

    if warning_count == 0:
        st.success("Dataset loaded and validation checks passed.")
    else:
        st.warning(f"Dataset loaded with {warning_count} validation warning(s).")

    with st.expander("Dataset details", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Series length", len(full_series))
        col2.metric("Start", month_label(full_series.index.min()))
        col3.metric("End", month_label(full_series.index.max()))

        metadata = load_result.meta or {}
        if metadata:
            st.write(
                {
                    "TITLE": metadata.get("TITLE"),
                    "CDID": metadata.get("CDID"),
                    "RELEASE DATE": metadata.get("RELEASE DATE"),
                    "NEXT RELEASE": metadata.get("NEXT RELEASE"),
                }
            )

        validation_warnings = getattr(validation_report, "warnings", [])
        if validation_warnings:
            st.write("Validation warnings:")
            st.write(validation_warnings)

    if not run_button:
        st.info("Select the analysis settings in the sidebar and click Run analysis.")
        st.stop()

    selected_origin_timestamp = pd.Timestamp(selected_origin_label)
    training_series = full_series.loc[:selected_origin_timestamp].copy()

    if training_series.empty:
        st.error("The selected origin month produced an empty training series.")
        st.stop()

    # -------------------------
    # Historical comparison mode
    # -------------------------
    if compare_methods and analysis_mode == "Historical evaluation":
        comparison_method_keys = list(METHODS.keys())
        forecast_results: dict[str, pd.DataFrame] = {}
        evaluation_results: dict[str, pd.DataFrame] = {}
        metrics_by_method: dict[str, dict[str, float]] = {}

        try:
            for method_key in comparison_method_keys:
                forecast_df = METHODS[method_key](training_series, h=forecast_horizon).copy()
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                forecast_df = forecast_df.sort_values("date").reset_index(drop=True)
                forecast_results[method_key] = forecast_df

                evaluation_df, metrics = evaluate_single_run(full_series, forecast_df)
                evaluation_results[method_key] = evaluation_df
                metrics_by_method[method_key] = metrics
        except Exception as exc:
            st.error(f"Comparison run failed: {exc}")
            st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("Comparison", f"{len(comparison_method_keys)} methods")
        col2.metric("Origin month", month_label(training_series.index.max()))
        col3.metric("Horizon", f"{forecast_horizon} months")

        st.caption(
            f"Historical comparison generated from origin {month_label(training_series.index.max())} "
            f"and evaluated against realised outcomes over the next {forecast_horizon} month(s)."
        )

        st.subheader("Historical forecast comparison")

        comparison_plot_df = build_comparison_plot_df(
            train_series=training_series,
            full_series=full_series,
            forecast_results=forecast_results,
            history_tail=history_tail,
        )

        comparison_chart = (
            alt.Chart(comparison_plot_df.dropna(subset=["value"]))
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X(
                    "date:T",
                    title="Month",
                    axis=alt.Axis(format="%Y-%m", labelAngle=-35),
                ),
                y=alt.Y("value:Q", title="PSNB"),
                color=alt.Color("Series:N", title="Series"),
            )
            .properties(height=420)
        )

        origin_rule = (
            alt.Chart(pd.DataFrame({"date": [training_series.index.max()]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(x="date:T")
        )

        st.altair_chart(comparison_chart + origin_rule, use_container_width=True)

        st.caption(
            "The chart compares all active forecast methods from the same historical origin "
            "against the realised values that later occurred."
        )

        st.subheader("Comparison metrics")
        comparison_metrics_table = build_comparison_metrics_table(metrics_by_method)
        st.dataframe(comparison_metrics_table, use_container_width=True)

        comparison_export_frames = []

        for method_key in comparison_method_keys:
            evaluation_df = evaluation_results[method_key].copy()
            evaluation_df["method_label"] = METHOD_LABELS.get(method_key, method_key)
            comparison_export_frames.append(evaluation_df)

        comparison_export_df = pd.concat(comparison_export_frames, ignore_index=True)
        comparison_export_df = comparison_export_df.sort_values(
            ["method_label", "horizon"]
        ).reset_index(drop=True)

        comparison_export_df = format_table_dates(
            comparison_export_df[
                [
                    "date",
                    "method",
                    "method_label",
                    "horizon",
                    "actual",
                    "point",
                    "lo80",
                    "hi80",
                    "lo95",
                    "hi95",
                    "abs_error",
                    "sq_error",
                    "hit80",
                    "hit95",
                    "width80",
                    "width95",
                ]
            ]
        )

        st.download_button(
            label="Download comparison CSV",
            data=df_to_csv_bytes(comparison_export_df),
            file_name=f"comparison_all_methods_{selected_origin_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Detailed comparison table", expanded=False):
            st.dataframe(comparison_export_df, use_container_width=True)

        best_mae_row = comparison_metrics_table.sort_values("MAE", ascending=True).iloc[0]
        best_rmse_row = comparison_metrics_table.sort_values("RMSE", ascending=True).iloc[0]

        st.caption(
            f"On this selected origin and horizon, the lowest MAE was achieved by "
            f"{best_mae_row['Method']} and the lowest RMSE was achieved by "
            f"{best_rmse_row['Method']}."
        )

        st.stop()

    # -------------------------
    # Single-method forecast flow
    # -------------------------
    forecast_function = METHODS[selected_method_key]
    selected_method_label = METHOD_LABELS.get(selected_method_key, selected_method_key)

    try:
        forecast_df = forecast_function(training_series, h=forecast_horizon).copy()
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Method", selected_method_label)
    col2.metric("Origin month", month_label(training_series.index.max()))
    col3.metric("Horizon", f"{forecast_horizon} months")

    if analysis_mode == "Forecast generation":
        st.caption(
            f"Forecast generated using data available up to {month_label(training_series.index.max())}."
        )
    else:
        st.caption(
            f"Historical evaluation generated from origin {month_label(training_series.index.max())} "
            f"and compared with realised outcomes over the next {forecast_horizon} month(s)."
        )

    forecast_chart_bytes = render_plot_image(
        history=training_series if analysis_mode == "Historical evaluation" else full_series,
        forecast_df=forecast_df,
        title=f"PSNB Forecast ({selected_method_label})",
        history_tail=history_tail,
        meta=load_result.meta,
        method_label=selected_method_label,
    )

    # In historical evaluation mode, keep the two-chart top layout:
    # left = forecast chart, right = realised outcomes for the same horizon.
    if analysis_mode == "Historical evaluation":
        left_chart_col, right_chart_col = st.columns(2)

        with left_chart_col:
            st.subheader("Forecast chart")
            st.image(forecast_chart_bytes, use_container_width=True)
            st.caption(
                "Forecast generated from the selected historical origin using only the data "
                "available at that time."
            )

        with right_chart_col:
            st.subheader("What actually happened")
            actual_window_df = build_actual_window_plot_df(full_series, forecast_df)

            if actual_window_df.empty:
                st.info("No realised values were available for the selected forecast window.")
            else:
                actual_chart = (
                    alt.Chart(actual_window_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X(
                            "date:T",
                            title="Month",
                            axis=alt.Axis(format="%Y-%m", labelAngle=-35),
                        ),
                        y=alt.Y("Actual:Q", title="PSNB"),
                    )
                    .properties(height=420)
                )
                st.altair_chart(actual_chart, use_container_width=True)
                st.caption(
                    "These are the real observed values from the original dataset for the same "
                    "future months covered by the historical forecast."
                )
    else:
        st.image(forecast_chart_bytes, use_container_width=True)

    download_col1, download_col2 = st.columns(2)

    download_col1.download_button(
        label="Download forecast CSV",
        data=df_to_csv_bytes(format_table_dates(forecast_df)),
        file_name=f"forecast_{selected_method_key}_{selected_origin_label}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    download_col2.download_button(
        label="Download chart PNG",
        data=forecast_chart_bytes,
        file_name=f"forecast_{selected_method_key}_{selected_origin_label}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.subheader("Forecast output")
    st.dataframe(format_table_dates(forecast_df), use_container_width=True)

    # -------------------------
    # Single-method historical evaluation
    # -------------------------
    if analysis_mode == "Historical evaluation":
        st.subheader("Historical evaluation against realised outcomes")

        evaluation_df, metrics = evaluate_single_run(full_series, forecast_df)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("MAE", f"{metrics['mae']:.2f}" if not np.isnan(metrics["mae"]) else "N/A")
        metric_col2.metric("RMSE", f"{metrics['rmse']:.2f}" if not np.isnan(metrics["rmse"]) else "N/A")
        metric_col3.metric("Actuals available", metrics["n_actual"])

        metric_col4, metric_col5, metric_col6, metric_col7 = st.columns(4)
        metric_col4.metric(
            "Coverage 80%",
            f"{metrics['coverage80']:.3f}" if not np.isnan(metrics["coverage80"]) else "N/A",
        )
        metric_col5.metric(
            "Coverage 95%",
            f"{metrics['coverage95']:.3f}" if not np.isnan(metrics["coverage95"]) else "N/A",
        )
        metric_col6.metric(
            "Average width 80%",
            f"{metrics['avg_width80']:.2f}" if not np.isnan(metrics["avg_width80"]) else "N/A",
        )
        metric_col7.metric(
            "Average width 95%",
            f"{metrics['avg_width95']:.2f}" if not np.isnan(metrics["avg_width95"]) else "N/A",
        )

        st.write("Forecast versus realised outcomes")

        evaluation_plot_df = build_evaluation_plot_df(
            train_series=training_series,
            full_series=full_series,
            forecast_df=forecast_df,
            history_tail=history_tail,
        )

        line_source = evaluation_plot_df.melt(
            id_vars="date",
            value_vars=["History", "Forecast", "Actual"],
            var_name="Series",
            value_name="value",
        ).dropna(subset=["value"])

        interval_source = evaluation_plot_df.dropna(subset=["Lo80", "Hi80"]).copy()

        interval_band = (
            alt.Chart(interval_source)
            .mark_area(opacity=0.15)
            .encode(
                x=alt.X("date:T", title="Month"),
                y=alt.Y("Lo80:Q", title="PSNB"),
                y2="Hi80:Q",
            )
        )

        line_chart = (
            alt.Chart(line_source)
            .mark_line()
            .encode(
                x=alt.X(
                    "date:T",
                    title="Month",
                    axis=alt.Axis(format="%Y-%m", labelAngle=-35),
                ),
                y=alt.Y("value:Q", title="PSNB"),
                color=alt.Color("Series:N", title="Series"),
            )
        )

        origin_rule = (
            alt.Chart(pd.DataFrame({"date": [training_series.index.max()]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(x="date:T")
        )

        evaluation_chart = (interval_band + line_chart + origin_rule).properties(height=420)
        st.altair_chart(evaluation_chart, use_container_width=True)

        st.caption(
            "The chart shows the observed series up to the selected evaluation origin, "
            "the forecast generated from that point, the realised values that later occurred, "
            "and the 80% prediction interval."
        )

        summary_columns = ["date", "horizon", "point", "actual", "abs_error", "hit80", "hit95"]
        st.dataframe(
            format_table_dates(evaluation_df[summary_columns]),
            use_container_width=True,
        )

        evaluation_download_df = format_table_dates(
            evaluation_df[
                [
                    "date",
                    "method",
                    "horizon",
                    "actual",
                    "point",
                    "lo80",
                    "hi80",
                    "lo95",
                    "hi95",
                    "abs_error",
                    "sq_error",
                    "hit80",
                    "hit95",
                    "width80",
                    "width95",
                ]
            ]
        )

        st.download_button(
            label="Download evaluation CSV",
            data=df_to_csv_bytes(evaluation_download_df),
            file_name=f"evaluation_{selected_method_key}_{selected_origin_label}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Detailed evaluation table", expanded=False):
            st.dataframe(evaluation_download_df, use_container_width=True)

        st.caption(
            "This mode performs a single historical holdout test: the model is trained only on data "
            "available up to the selected origin month, then compared with the realised values that followed."
        )


if __name__ == "__main__":
    main()