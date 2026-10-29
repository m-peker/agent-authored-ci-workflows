"""Precision-corrected flag rates from the two-rater human validation.

For each category (breaking, security_high) and author group, the human precision of a sampled finding
(TP / (TP+FP); UNSURE excluded, or counted as FP in a sensitivity variant) is multiplied by the observed
file-level flag rate (D1, results/v3/robust_pooled.csv) to estimate the rate of files with a confirmed problem.
Because the sample took one finding per file, finding precision approximates the probability that a flagged file
has at least one true problem from below, so corrected rates are conservative (lower) estimates.
95% CIs by bootstrap over the annotated items (10,000 draws, seed 42), holding the observed rate fixed (its
sampling error is small relative to the 30-item precision estimates).
Output: results/human_precision_corrected.csv
"""
import re

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
B = 10_000
D = "data/annotation/human"


def load(f):
    d = {}
    for line in open(f, encoding="utf-8-sig"):
        m = re.match(r"^([FP]-[A-Z]?\d+)\s*:\s*([A-Za-z]+)", line.strip())
        if m:
            d[m.group(1)] = m.group(2).upper()
    return d


def main():
    h1, h2 = load(f"{D}/hakem1.txt"), load(f"{D}/hakem2.txt")
    k = pd.read_csv("data/annotation/key.csv")
    k["id"] = "F-" + k.item_id
    k["R1"], k["R2"] = k.id.map(h1), k.id.map(h2)
    k["consensus"] = np.where(k.R1 == k.R2, k.R1, "UNSURE")
    rates = pd.read_csv("results/v3/robust_pooled.csv")
    rates = rates[rates.design == "D1_first_draft"].set_index("outcome")
    rows = []
    for rater in ("consensus", "R1", "R2"):
        for variant in ("unsure_excluded", "unsure_as_fp"):
            for cat in ("breaking", "security_high"):
                boots, point = {}, {}
                for grp, col in (("agent", "rate_agent"), ("human", "rate_human")):
                    lab = k[(k.cat == cat) & (k.group == grp)][rater].values
                    if variant == "unsure_excluded":
                        lab = lab[lab != "UNSURE"]
                    tp = (lab == "TP").astype(float)
                    rate = rates.loc[cat, col]
                    point[grp] = rate * tp.mean()
                    boots[grp] = rate * tp[RNG.integers(0, len(tp), (B, len(tp)))].mean(1)
                    lo, hi = np.percentile(boots[grp], [2.5, 97.5])
                    rows.append({"rater": rater, "variant": variant, "category": cat, "group": grp, "n": len(tp),
                                 "precision": tp.mean(), "raw_rate": rate, "corrected_rate": point[grp],
                                 "lo": lo, "hi": hi})
                diff = boots["agent"] - boots["human"]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                rows.append({"rater": rater, "variant": variant, "category": cat, "group": "agent-human",
                             "n": np.nan, "precision": np.nan, "raw_rate": rates.loc[cat, "rate_agent"] -
                             rates.loc[cat, "rate_human"], "corrected_rate": point["agent"] - point["human"],
                             "lo": lo, "hi": hi})
    out = pd.DataFrame(rows).round(3)
    out.to_csv("results/human_precision_corrected.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
