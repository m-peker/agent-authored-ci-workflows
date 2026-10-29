"""Power and equivalence for the agent-vs-human comparisons (reads robustness.py outputs; CPU only).

For estimates reported as percentage-point differences (E2 within-repo FE-LPM, E3 CEM):
  SE        = (CI_high - CI_low) / (2 * 1.96)
  MDE80     = (1.96 + 0.8416) * SE        smallest true difference detectable with 80% power, alpha .05 two-sided
  TOST      = two one-sided tests against an equivalence margin +-m (m = 3, 5 pp);
              equivalent at alpha .05 iff the 90% CI lies inside (-m, +m). p_tost = max of the two one-sided p's.
For E1 (GEE odds ratio): SE on the log-OR scale, MDE80 reported as an odds ratio.
A "no difference" claim is made only when TOST rejects non-equivalence; otherwise the result is "inconclusive".
Outputs: results/power_pooled.csv, results/power_by_agent.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

Z, ZP = 1.959964, 0.841621
MARGINS = [3, 5]
SRC = sys.argv[1] if len(sys.argv) > 1 else "results"


def add(df, est, lo, hi, tag):
    se = (df[hi] - df[lo]) / (2 * Z)
    df[f"{tag}_se"] = se
    df[f"{tag}_mde80_pp"] = (Z + ZP) * se
    for m in MARGINS:
        p_low = 1 - norm.cdf((df[est] + m) / se)   # H0: diff <= -m
        p_high = norm.cdf((df[est] - m) / se)      # H0: diff >= +m
        df[f"{tag}_tost{m}_p"] = np.maximum(p_low, p_high)
        df[f"{tag}_equiv{m}"] = df[f"{tag}_tost{m}_p"] < 0.05
    return df


def verdict(r, tag):
    if pd.isna(r.get(f"{tag}_se")):
        return "n/a"
    if r[f"{tag}_lo"] > 0 or r[f"{tag}_hi"] < 0:  # 95% CI excludes zero
        return "different"
    if r[f"{tag}_equiv3"]:
        return "equivalent(+-3pp)"
    if r[f"{tag}_equiv5"]:
        return "equivalent(+-5pp)"
    return "inconclusive"


for name, keys in [("pooled", ["design", "outcome"]), ("by_agent", ["design", "outcome", "agent"])]:
    d = pd.read_csv(f"{SRC}/robust_{name}.csv")
    if "E2_lo" not in d:  # by-agent table stores only the point estimate + p for E2 -> derive CI from p
        se = np.abs(d.E2_diff_pp) / np.maximum(norm.isf(d.E2_p / 2), 1e-9)
        d["E2_lo"], d["E2_hi"] = d.E2_diff_pp - Z * se, d.E2_diff_pp + Z * se
    d = add(d, "E2_diff_pp", "E2_lo", "E2_hi", "E2")
    d = add(d, "E3_diff_pp", "E3_lo", "E3_hi", "E3")
    if "E1_lo" in d:
        lse = (np.log(d.E1_hi) - np.log(d.E1_lo)) / (2 * Z)
        d["E1_mde80_OR"] = np.exp((Z + ZP) * lse)
    d["E2_verdict"] = d.apply(lambda r: verdict(r, "E2"), axis=1)
    d["E3_verdict"] = d.apply(lambda r: verdict(r, "E3"), axis=1)
    d.round(4).to_csv(f"{SRC}/power_{name}.csv", index=False)
    cols = keys + [c for c in ["E2_diff_pp", "E2_mde80_pp", "E2_tost5_p", "E2_verdict",
                               "E3_diff_pp", "E3_mde80_pp", "E3_tost5_p", "E3_verdict", "E1_mde80_OR"] if c in d]
    print(f"\n== {name}\n", d[cols].round(3).to_string(index=False))
