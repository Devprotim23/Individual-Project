from ons_io import load_series  # only if your file is literally named ons_io.py in the same folder
from validate import validate_monthly_series
# Better later: from psnb.io import load_series (once you move into src/psnb)

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
