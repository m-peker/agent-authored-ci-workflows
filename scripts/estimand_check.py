"""Why do E2 (within-repo FE) and E3 (matched, agent-weighted) disagree for D1 breaking?

Both adjust for repo and size; they differ in how strata (repo x size quintile) are weighted:
  ATT      weight = n_agent in stratum            (E3 as reported: effect where agents actually work)
  ATC      weight = n_human in stratum
  overlap  weight = n_a * n_h / (n_a + n_h)        (≈ what the FE regression targets)
  equal    weight = 1 per stratum
Plus concentration checks for ATT: drop the top-10 repos by agent file count, and cap each repo's weight.
And E3 with size quintiles computed within repo instead of globally.
All CIs: repo-cluster bootstrap, 1,000 reps, seed 42.
File sizes come from data/file_sizes.json (read from the backup archive, so files need not be extracted).
Output: results/estimand_check.csv
"""
import json
import os

import numpy as np
import pandas as pd

import robustness as rb

SIZES = json.load(open("data/file_sizes.json"))
rb.os.path.getsize = lambda p: SIZES[p.replace(os.sep, "/")]


def strata(base, within_repo=False):
    if within_repo:
        q = base.groupby("repo").log_size.transform(
            lambda s: pd.qcut(s.rank(method="first"), min(5, len(s)), labels=False) if len(s) > 1 else 0)
    else:
        q = pd.qcut(base.log_size, 5, labels=False, duplicates="drop")
    b = base.assign(stratum=base.repo + "|" + q.astype(str))
    g = b.groupby(["stratum", "agent_any"]).y.agg(["mean", "size"]).unstack().dropna()
    out = pd.DataFrame({"diff": g[("mean", 1)] - g[("mean", 0)], "na": g[("size", 1)], "nh": g[("size", 0)]})
    out["repo"] = [s.split("|")[0] for s in out.index]
    return out


WEIGHTS = {
    "ATT": lambda s: s.na,
    "ATC": lambda s: s.nh,
    "overlap": lambda s: s.na * s.nh / (s.na + s.nh),
    "equal": lambda s: pd.Series(1.0, index=s.index),
}


def est_ci(s, wfun, rng):
    w = wfun(s).values.astype(float)
    dv, repos = s["diff"].values, s.repo.values
    est = 100 * np.average(dv, weights=w)
    ur = np.unique(repos)
    idx = {r: np.where(repos == r)[0] for r in ur}
    boots = []
    for _ in range(1000):
        pick = np.concatenate([idx[r] for r in rng.choice(ur, len(ur))])
        boots.append(100 * np.average(dv[pick], weights=w[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def main():
    d = rb.build()
    rows = []
    for design in ["D1_first_draft", "D2_final_merged"]:
        for y in ["breaking", "security_high", "supply_chain"]:
            base = d[(d.design == design) & d[y].notna()].copy()
            base["y"] = base[y].astype(int)
            fe = rb.fe_lpm(base)
            rows.append({"design": design, "outcome": y, "estimator": "E2_FE", "est": fe["E2_diff_pp"],
                         "lo": fe["E2_lo"], "hi": fe["E2_hi"], "repos": fe["E2_repos"]})
            for within in (False, True):
                s = strata(base, within)
                tag = "withinQ" if within else "globalQ"
                for name, wf in WEIGHTS.items():
                    e, lo, hi = est_ci(s, wf, np.random.default_rng(42))
                    rows.append({"design": design, "outcome": y, "estimator": f"E3_{name}_{tag}", "est": e,
                                 "lo": lo, "hi": hi, "repos": s.repo.nunique()})
                if not within:
                    top = s.groupby("repo").na.sum().nlargest(10).index
                    share = s.groupby("repo").na.sum().nlargest(10).sum() / s.na.sum()
                    e, lo, hi = est_ci(s[~s.repo.isin(top)], WEIGHTS["ATT"], np.random.default_rng(42))
                    rows.append({"design": design, "outcome": y, "estimator": "E3_ATT_drop_top10_repos", "est": e,
                                 "lo": lo, "hi": hi, "repos": s.repo.nunique() - 10,
                                 "note": f"top10 repos = {100 * share:.0f}% of agent weight"})
                    cap = s.groupby("repo").na.transform("sum")
                    e, lo, hi = est_ci(s, lambda t: t.na / cap.loc[t.index], np.random.default_rng(42))
                    rows.append({"design": design, "outcome": y, "estimator": "E3_ATT_repo_equal", "est": e,
                                 "lo": lo, "hi": hi, "repos": s.repo.nunique()})
            print(design, y, "done", flush=True)
    r = pd.DataFrame(rows).round(2)
    r["sig"] = (r.lo > 0) | (r.hi < 0)
    r.to_csv("results/estimand_check.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
