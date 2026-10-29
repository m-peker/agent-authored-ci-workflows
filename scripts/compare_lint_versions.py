"""Tool-version sensitivity: v1 (actionlint 1.7.7 / zizmor 1.5.2) vs v2 (1.7.12 / 1.31.1) lint outputs."""
import pandas as pd

for t in ["", "_human"]:
    a = pd.read_parquet(f"data/processed/v1/findings{t}.parquet")
    b = pd.read_parquet(f"data/processed/findings{t}.parquet")
    la = pd.read_parquet(f"data/processed/v1/linted_files{t}.parquet")
    lb = pd.read_parquet(f"data/processed/linted_files{t}.parquet")
    print(f"\n== {t or 'agent'}: linted v1 {len(la):,} v2 {len(lb):,} | findings v1 {len(a):,} v2 {len(b):,}")
    print(pd.concat([a.groupby("tool").size().rename("v1"), b.groupby("tool").size().rename("v2")], axis=1).to_string())
    print("zizmor input errors v1/v2:", (a.rule == "_input_error").sum(), (b.rule == "_input_error").sum())
    top = pd.concat([a.groupby(["tool", "rule"]).size().rename("v1"), b.groupby(["tool", "rule"]).size().rename("v2")], axis=1)
    print(top.fillna(0).astype(int).sort_values("v2", ascending=False).head(15).to_string())
