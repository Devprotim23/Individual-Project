from ons_io import load_series  # only if your file is literally named ons_io.py in the same folder
# Better later: from psnb.io import load_series (once you move into src/psnb)

res = load_series("series-050326.csv")
s = res.series

print(s.head())
print(s.tail())
print("n =", len(s))
print("meta keys =", list(res.meta.keys()))