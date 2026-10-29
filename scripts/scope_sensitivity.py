"""Does the agent-human result survive removing non-workflow files, agent-product templates and copies?

Samples (newly added files; human side filtered the same way):
  S0_all            every added *.yml|yaml under any .github/workflows path (the earlier definition)
  S1_active         only workflows GitHub runs (.github/workflows/<file> at the repo root)  -> new main sample
  S2_no_template    S1 without agent-product integration files (copilot-setup-steps, claude-code-action, ...)
  S3_dedup          S1 keeping one file per normalised-content hash and group (copies / boilerplate collapsed)
  S4_strict         S2 and S3 combined
  P_<purpose>       S2 within one workflow purpose (agent vs human of the same kind)
  A_<agent>         S2, one agent vs humans
Per sample x design x outcome: rates, E1 GEE OR, E2 within-repo FE diff (TOST at +-3/+-5 pp from its CI),
E3 matched ATT and E3 with repos weighted equally. Holm across the P_ and A_ families separately.
Outputs: results/scope_sensitivity.csv, results/scope_composition.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

os.environ["SCOPE"] = "all"
import robustness as rb  # noqa: E402

OUTS = ["breaking", "security_high", "supply_chain"]
RNG = np.random.default_rng(42)


def e3(base, repo_equal=False):
    q = pd.qcut(base.log_size, 5, labels=False, duplicates="drop").astype(str)
    b = base.assign(stratum=base.repo + "|" + q)
    g = b.groupby(["stratum", "agent_any"]).y.agg(["mean", "size"]).unstack().dropna()
    if len(g) < 10:
        return np.nan, np.nan, np.nan
    diff = (g[("mean", 1)] - g[("mean", 0)]).values
    repos = np.array([s.split("|")[0] for s in g.index])
    w = g[("size", 1)].values.astype(float)
    if repo_equal:
        tot = pd.Series(w).groupby(repos).transform("sum").values
        w = w / tot
    est = 100 * np.average(diff, weights=w)
    ur = np.unique(repos)
    idx = {r: np.where(repos == r)[0] for r in ur}
    bs = []
    for _ in range(1000):
        pick = np.concatenate([idx[r] for r in RNG.choice(ur, len(ur))])
        bs.append(100 * np.average(diff[pick], weights=w[pick]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def tost_verdict(est, lo, hi):
    if np.isnan(est):
        return "n/a"
    se = (hi - lo) / (2 * 1.96)
    for m in (3, 5):
        p = max(norm.sf((est + m) / se), norm.cdf((est - m) / se))
        if p < 0.05:
            return f"equivalent(+-{m})"
    return "different" if (lo > 0 or hi < 0) else "inconclusive"


def estimate(base, label, design, y):
    base = base[base[y].notna()].copy()
    base["y"] = base[y].astype(int)
    na, nh = int(base.agent_any.sum()), int((1 - base.agent_any).sum())
    r = {"sample": label, "design": design, "outcome": y, "n_human": nh, "n_agent": na,
         "rate_human": 100 * base[base.agent_any == 0].y.mean(), "rate_agent": 100 * base[base.agent_any == 1].y.mean()}
    if nh < 30 or na < 30:
        return r
    try:
        m = rb.gee(base, "y ~ agent_any + log_size")
        r.update(E1_OR=np.exp(m.params.agent_any), E1_p=m.pvalues.agent_any)
    except Exception:
        pass
    fe = rb.fe_lpm(base)
    if fe:
        r.update(E2=fe["E2_diff_pp"], E2_lo=fe["E2_lo"], E2_hi=fe["E2_hi"], E2_p=fe["E2_p"], E2_repos=fe["E2_repos"],
                 E2_verdict=tost_verdict(fe["E2_diff_pp"], fe["E2_lo"], fe["E2_hi"]))
    r["E3_ATT"], r["E3_ATT_lo"], r["E3_ATT_hi"] = e3(base)
    r["E3_repoeq"], r["E3_repoeq_lo"], r["E3_repoeq_hi"] = e3(base, repo_equal=True)
    return r


def main():
    d = rb.build()
    d = d[d.design.isin(["D1_first_draft", "D2_final_merged"])]
    # composition
    comp = []
    for design, g in d.groupby("design"):
        for grp, gg in g.groupby(d.agent_any.map({1: "agent", 0: "human"})):
            comp.append({"design": design, "group": grp, "files": len(gg), "repos": gg.repo.nunique(),
                         "active_pct": 100 * gg.active.mean(),
                         "template_pct_of_active": 100 * gg[gg.active].template.mean(),
                         "dup_files_pct_of_active": 100 * gg[gg.active].duplicated("nhash").mean(),
                         **{f"purpose_{k}": 100 * v for k, v in
                            gg[gg.active].purpose.value_counts(normalize=True).items()}})
    comp = pd.DataFrame(comp).round(1)
    comp.to_csv("results/scope_composition.csv", index=False)
    print(comp.T.to_string())

    samples = {}
    s1 = d[d.active]
    samples["S0_all"] = d
    samples["S1_active"] = s1
    samples["S2_no_template"] = s1[~s1.template]
    dedup = lambda x: x.drop_duplicates(["design", "agent_any", "nhash"])  # noqa: E731
    samples["S3_dedup"] = dedup(s1)
    samples["S4_strict"] = dedup(s1[~s1.template])
    s2 = samples["S2_no_template"]
    for p in sorted(s2.purpose.unique()):
        samples[f"P_{p}"] = s2[s2.purpose == p]
    for a in sorted(set(s2.group) - {"Human"}):
        samples[f"A_{a}"] = s2[s2.group.isin(["Human", a])]

    rows = []
    for label, sd in samples.items():
        for design in ["D1_first_draft", "D2_final_merged"]:
            base = sd[sd.design == design]
            for y in OUTS:
                rows.append(estimate(base, label, design, y))
        print("done", label, flush=True)
    r = pd.DataFrame(rows)
    for fam in ("P_", "A_"):
        for col in ("E1_p", "E2_p"):
            m = r["sample"].str.startswith(fam) & r[col].notna()
            if m.any():
                r.loc[m, col + "_holm"] = multipletests(r.loc[m, col], method="holm")[1]
    r = r.round(3)
    r.to_csv("results/scope_sensitivity.csv", index=False)
    cols = ["sample", "design", "outcome", "n_human", "n_agent", "rate_human", "rate_agent", "E1_OR", "E2", "E2_lo",
            "E2_hi", "E2_verdict", "E3_ATT", "E3_repoeq"]
    print(r[[c for c in cols if c in r]].to_string(index=False))


if __name__ == "__main__":
    main()
