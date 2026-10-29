"""Which zizmor audits drive security_high in v2 (per-file prevalence among agent-added first versions)."""
import pandas as pd

v = pd.read_parquet("data/processed/versions.parquet")
v = v[(v.origin == "added") & (v.ver == 0)][["sha", "filename"]]
f = pd.read_parquet("data/processed/findings.parquet").merge(v, on=["sha", "filename"])
z = f[f.tool == "zizmor"]
n = len(v)
t = (z.groupby(["rule", "severity"]).apply(lambda g: g[["sha", "filename"]].drop_duplicates().shape[0])
       .rename("files").reset_index())
t["pct_files"] = (100 * t.files / n).round(1)
print(f"agent-added first versions: {n:,}")
print(t.sort_values("files", ascending=False).head(40).to_string(index=False))
