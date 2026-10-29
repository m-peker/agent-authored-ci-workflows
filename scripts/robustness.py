"""Robustness battery for the agent-vs-human comparison (RQ1). Newly ADDED workflow files only.

Designs (which agent version is compared with the human commit):
  D1_first_draft  : agent's first version of the file (ver == 0), all PRs
  D2_final_merged : agent's last version of the file in a MERGED PR — the closest analogue of a human
                    commit on the default branch (both are the state that actually landed)
Outcomes:
  breaking        : actionlint correctness finding or unparseable (static)
  security_high   : zizmor High severity, High/Medium confidence, excluding pinning audits (static)
  supply_chain    : action/image not pinned to a digest (static; see categories.py)
  ci_failure      : any completed run of that file at that SHA failed (versions with >= 1 run only)
Estimators:
  E1_gee          : GEE logistic, exchangeable within repo, + log(size)            [population-level]
  E2_within_repo  : linear probability model with repo fixed effects, + log(size), repo-clustered SE
                    -> agent-human difference (pp) inside the same repository
  E3_cem          : coarsened exact matching on repo x size quintile; agent-weighted rate difference (pp)
                    with repo-cluster bootstrap 95% CI (1,000 reps, seed 42)
By-agent p-values are Holm-adjusted within each design x outcome x estimator.
Outputs: results/robust_pooled.csv, results/robust_by_agent.csv, results/robust_cursor_diag.csv
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
from categories import flag_frame
import wf_meta
OUTCOMES = ["breaking", "security_high", "supply_chain", "ci_failure"]


def static_flags(find):
    return flag_frame(find)[["sha", "filename", "breaking", "security_high", "supply_chain"]]


def load_runs():
    recs, fetched = [], set()
    for f in glob.glob("data/runs/*.json"):
        d = json.load(open(f))
        fetched.add(d["sha"])
        recs += [{"sha": d["sha"], "filename": r["path"], "conclusion": r["conclusion"], "event": r["event"]}
                 for r in d["runs"] if r["status"] == "completed"]
    r = pd.DataFrame(recs)
    per = (r.groupby(["sha", "filename"])
             .agg(ci_failure=("conclusion", lambda s: s.isin(["failure", "startup_failure"]).any()),
                  event=("event", lambda s: s.mode().iat[0]))
             .reset_index())
    return per, fetched


def build():
    h = pd.read_parquet("data/processed/human_wf_changes.parquet")
    h = h[h.status == "added"].drop_duplicates(["sha", "filename"])
    h = h.merge(pd.read_parquet("data/processed/linted_files_human.parquet"), on=["sha", "filename"])
    h = h.merge(static_flags(pd.read_parquet("data/processed/findings_human.parquet")),
                on=["sha", "filename"], how="left")
    h["group"], h["root"] = "Human", "data/files_human"

    v = pd.read_parquet("data/processed/versions.parquet")
    v = v[v.origin == "added"].copy()
    v["repo"] = v.repo_url.str.replace("https://api.github.com/repos/", "", regex=False)
    v["breaking"] = v.ci_correctness | v.unparseable
    v["group"], v["root"] = v.agent.astype(str), "data/files"

    cols = ["repo", "sha", "filename", "group", "root", "breaking", "security_high", "supply_chain"]
    d1 = pd.concat([h[cols], v[v.ver == 0][cols]], ignore_index=True).assign(design="D1_first_draft")
    final = v[(v.ver == v.n_ver - 1) & v.merged]
    d2 = pd.concat([h[cols], final[cols]], ignore_index=True).assign(design="D2_final_merged")
    d = pd.concat([d1, d2], ignore_index=True)
    d[["breaking", "security_high", "supply_chain"]] = d[["breaking", "security_high", "supply_chain"]].fillna(False).astype(bool)
    d["size"] = [os.path.getsize(os.path.join(r, s, f)) for r, s, f in zip(d.root, d.sha, d.filename)]
    d["log_size"] = np.log1p(d["size"])
    # file metadata (wf_meta.py): active workflow, agent-product template, purpose, normalised-content hash
    texts = [open(os.path.join(r, s, f), encoding="utf-8", errors="replace").read()
             for r, s, f in zip(d.root, d.sha, d.filename)]
    d["active"] = [wf_meta.active(f) for f in d.filename]
    d["purpose"] = [wf_meta.purpose(t, f) for t, f in zip(texts, d.filename)]
    d["template"] = d.purpose == "agent_integration"
    d["nhash"] = [wf_meta.nhash(t) for t in texts]
    # SCOPE: "active" (default) = only files GitHub runs; "all" = every added *.yml under any .github/workflows path
    if os.environ.get("SCOPE", "active") == "active":
        d = d[d.active].copy()

    per, fetched = load_runs()
    d = d.merge(per, on=["sha", "filename"], how="left")
    d.loc[~d.sha.isin(fetched), "ci_failure"] = np.nan  # run data not fetched for this SHA
    d["agent_any"] = (d.group != "Human").astype(int)
    return d


def gee(data, form):
    data = data.sort_values("repo")
    return smf.gee(form, groups="repo", data=data, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def fe_lpm(data):
    """Linear probability model with repo fixed effects (within transform), repo-clustered SE.
    Only repos containing both agent and human files identify the effect. Returns pp difference."""
    ok = data[data.groupby("repo").agent_any.transform("nunique") > 1]
    if ok.repo.nunique() < 10:
        return None
    w = ok[["y", "agent_any", "log_size"]].astype(float)
    w = w - w.groupby(ok.repo.values).transform("mean")
    m = sm.OLS(w.y, w[["agent_any", "log_size"]]).fit(cov_type="cluster", cov_kwds={"groups": ok.repo.values})
    ci = m.conf_int().loc["agent_any"]
    return {"E2_diff_pp": 100 * m.params.agent_any, "E2_lo": 100 * ci[0], "E2_hi": 100 * ci[1],
            "E2_p": m.pvalues.agent_any, "E2_repos": ok.repo.nunique()}


def cem_diff(data, agent_mask):
    """Agent-weighted mean of (agent rate - human rate) over repo x size-quintile strata with both."""
    data = data.assign(stratum=data.repo + "|" + pd.qcut(data.log_size, 5, labels=False, duplicates="drop").astype(str),
                       is_agent=agent_mask.astype(int))
    g = data.groupby(["stratum", "is_agent"]).y.agg(["mean", "size"]).unstack()
    g = g.dropna()
    if len(g) < 10:
        return np.nan, np.nan, np.nan, 0
    diff = g[("mean", 1)] - g[("mean", 0)]
    w = g[("size", 1)]
    est = 100 * np.average(diff, weights=w)
    # cluster bootstrap over strata' repos
    repos = np.array([s.split("|")[0] for s in g.index])
    ur = np.unique(repos)
    idx = {r: np.where(repos == r)[0] for r in ur}
    dv, wv, boots = diff.values, w.values, []
    for _ in range(1000):
        pick = np.concatenate([idx[r] for r in RNG.choice(ur, len(ur))])
        boots.append(100 * np.average(dv[pick], weights=wv[pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi, len(ur)


def main():
    d = build()
    agents = sorted(set(d.group) - {"Human"})
    pooled, by_agent = [], []
    for design in ["D1_first_draft", "D2_final_merged"]:
        for y in OUTCOMES:
            base = d[(d.design == design) & d[y].notna()].copy()
            base["y"] = base[y].astype(int)
            n_h, n_a = (base.group == "Human").sum(), (base.group != "Human").sum()
            rate_h = 100 * base[base.group == "Human"].y.mean()
            rate_a = 100 * base[base.group != "Human"].y.mean()
            row = {"design": design, "outcome": y, "n_human": n_h, "n_agent": n_a,
                   "rate_human": rate_h, "rate_agent": rate_a}
            m = gee(base, "y ~ agent_any + log_size")
            ci = m.conf_int().loc["agent_any"]
            row.update(E1_OR=np.exp(m.params.agent_any), E1_lo=np.exp(ci[0]), E1_hi=np.exp(ci[1]),
                       E1_p=m.pvalues.agent_any)
            row.update(fe_lpm(base) or {})
            est, lo, hi, nr = cem_diff(base, base.group != "Human")
            row.update(E3_diff_pp=est, E3_lo=lo, E3_hi=hi, E3_repos=nr)
            pooled.append(row)
            print(f"{design:16s} {y:14s} human {rate_h:5.1f}% (n={n_h}) agent {rate_a:5.1f}% (n={n_a}) | "
                  f"GEE OR {row['E1_OR']:.2f} p={row['E1_p']:.3g} | within-repo {row.get('E2_diff_pp', np.nan):+.1f}pp "
                  f"p={row.get('E2_p', np.nan):.3g} (repos={row.get('E2_repos', 0)}) | CEM {est:+.1f}pp [{lo:+.1f},{hi:+.1f}]", flush=True)

            # by agent: each agent vs Human separately (keeps estimators comparable)
            rows = []
            for ag in agents:
                sub = base[base.group.isin(["Human", ag])].copy()
                if (sub.group == ag).sum() < 30:
                    continue
                r = {"design": design, "outcome": y, "agent": ag, "n_agent": int((sub.group == ag).sum()),
                     "rate_agent": 100 * sub[sub.group == ag].y.mean(), "rate_human": rate_h}
                try:
                    m = gee(sub, "y ~ agent_any + log_size")
                    r.update(E1_OR=np.exp(m.params.agent_any), E1_p=m.pvalues.agent_any)
                except Exception:
                    pass
                r.update(fe_lpm(sub) or {})
                est, lo, hi, nr = cem_diff(sub, sub.group == ag)
                r.update(E3_diff_pp=est, E3_lo=lo, E3_hi=hi, E3_repos=nr)
                rows.append(r)
            ba = pd.DataFrame(rows)
            for e in ["E1", "E2"]:
                if f"{e}_p" in ba:
                    mask = ba[f"{e}_p"].notna()
                    ba.loc[mask, f"{e}_p_holm"] = multipletests(ba.loc[mask, f"{e}_p"], method="holm")[1]
            by_agent.append(ba)

    pd.DataFrame(pooled).round(4).to_csv("results/robust_pooled.csv", index=False)
    ba = pd.concat(by_agent, ignore_index=True).round(4)
    ba.to_csv("results/robust_by_agent.csv", index=False)
    print("\nBy agent:\n", ba.to_string(index=False))

    # Cursor diagnostic: why is its raw breaking rate low but GEE OR > 1?
    x = d[(d.design == "D1_first_draft") & d.group.isin(["Human", "Cursor", "Claude_Code"])]
    diag = (x.groupby("group")
              .agg(files=("sha", "size"), repos=("repo", "nunique"), breaking_pct=("breaking", lambda s: 100 * s.mean()),
                   median_size=("size", "median"),
                   top_repo_share=("repo", lambda s: 100 * s.value_counts(normalize=True).iat[0]),
                   files_per_repo=("repo", lambda s: len(s) / s.nunique())))
    # repo-averaged rate (each repo weighs 1): what GEE-exchangeable implicitly moves toward
    diag["repo_avg_breaking_pct"] = x.groupby(["group", "repo"]).breaking.mean().groupby("group").mean() * 100
    diag.round(2).to_csv("results/robust_cursor_diag.csv")
    print("\nCursor diagnostic:\n", diag.round(2).to_string())


if __name__ == "__main__":
    main()
