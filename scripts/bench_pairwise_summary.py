"""Summarise the blind pairwise task-fulfilment judgements (bench_pairwise.py).

Per judge and candidate: share of tasks where the candidate wins / ties / loses against the commercial agent's
first draft (position-consistent wins only), Wilson 95% CIs, and an exact sign test of wins vs losses (Holm
within judge). Inter-judge agreement on the 3-way result (Cohen's kappa) and a consensus result (both judges agree,
else tie). Sanity row: the landed reference should not lose to the agent's draft.
Output: results/bench_pairwise.csv, results/bench_pairwise_agreement.csv
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def summarise(d, judge):
    rows = []
    for c, g in d.groupby("candidate"):
        n = len(g)
        w, l = (g.result == "cand").sum(), (g.result == "agent").sum()
        r = {"judge": judge, "candidate": c, "n": n, "win": 100 * w / n, "tie": 100 * (n - w - l) / n,
             "loss": 100 * l / n, "net": 100 * (w - l) / n,
             "sign_p": binomtest(int(w), int(w + l), 0.5).pvalue if w + l else 1.0,
             "position_consistent": 100 * g.consistent.mean()}
        r["win_lo"], r["win_hi"] = [100 * x for x in proportion_confint(w, n, method="wilson")]
        r["loss_lo"], r["loss_hi"] = [100 * x for x in proportion_confint(l, n, method="wilson")]
        rows.append(r)
    out = pd.DataFrame(rows)
    out["sign_p_holm"] = multipletests(out.sign_p, method="holm")[1]
    return out


def main():
    files = {os.path.basename(f)[:-6]: f for f in glob.glob("data/bench/pairwise/*.jsonl") if "_smoke" not in f}
    ds = {j: pd.read_json(f, lines=True) for j, f in files.items()}
    res = [summarise(d, j) for j, d in ds.items()]
    agree = []
    if len(ds) >= 2:
        (j1, d1), (j2, d2) = list(ds.items())[:2]
        m = d1.merge(d2, on=["candidate", "task_id"], suffixes=("_1", "_2"))
        agree.append({"pair": f"{j1} vs {j2}", "n": len(m), "agreement": (m.result_1 == m.result_2).mean(),
                      "kappa": kappa(m.result_1, m.result_2)})
        m["result"] = np.where(m.result_1 == m.result_2, m.result_1, "tie")
        m["consistent"] = m.consistent_1 & m.consistent_2
        res.append(summarise(m, "consensus"))
    out = pd.concat(res).round(3)
    out.to_csv("results/bench_pairwise.csv", index=False)
    pd.DataFrame(agree).round(3).to_csv("results/bench_pairwise_agreement.csv", index=False)
    print(out[["judge", "candidate", "n", "win", "tie", "loss", "net", "sign_p_holm", "position_consistent"]]
          .to_string(index=False))
    print(pd.DataFrame(agree).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
