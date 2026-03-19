from ons_io import load_series
from validate import validate_monthly_series
from models import forecast_seasonal_naive, forecast_ets, forecast_arima
from backtest import expanding_window_backtest
from metrics import compute_errors, compute_coverage

res = load_series("series-050326.csv")
s = res.series

print(s.head())
print(s.tail())
print("n =", len(s))
print("meta keys =", list(res.meta.keys()))

rep = validate_monthly_series(s, expected_anchor="MS", strict=True)

print("\nVALIDATION REPORT:")
print(rep)

print("\nWARNINGS:")
for w in rep.warnings:
    print("-", w)

forecast_df = forecast_seasonal_naive(s, h=12)

print("\nSEASONAL NAIVE FORECAST:")
print(forecast_df)

ets_df = forecast_ets(s, h=12)

print("\nETS FORECAST:")
print(ets_df)

arima_df = forecast_arima(s, h=12)

print("\nARIMA FORECAST:")
print(arima_df)

pred_df = expanding_window_backtest(
    s,
    h_max=12,
    methods={
        "seasonal_naive": forecast_seasonal_naive,
        "ets": forecast_ets,
        "arima": forecast_arima,
    },
    min_train=60,
)

print("\nBACKTEST OUTPUT:")
print(pred_df.head())
print(pred_df.tail())
print("backtest rows =", len(pred_df))

errors_df = compute_errors(pred_df)
coverage_df = compute_coverage(pred_df)

print("\nERROR METRICS:")
print(errors_df)

print("\nCOVERAGE METRICS:")
print(coverage_df)