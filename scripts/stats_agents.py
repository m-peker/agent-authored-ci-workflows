"""Statistical tests for agent differences (RQ2/RQ3 observational part).

Models are GEE logistic regressions with exchangeable correlation inside repositories
(files/PRs from the same repo are not independent). Reference agent = Claude_Code (largest group).

M1  breaking issue in agent-ADDED workflow, first version  ~ agent + log(size)
M2  high-signal security issue, same population           ~ agent + log(size)
M3  PR merged  ~ breaking at last version + security_high at last version + agent + log(#versions)
Also: chi-square test of independence + Cramer's V for M1/M2 outcomes across agents.
Outputs: results/stats_models.csv, results/stats_chi2.csv
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

REF = "Claude_Code"
v = pd.read_parquet("data/processed/versions.parquet")
v["breaking"] = v.ci_correctness | v.unparseable
v["size"] = [os.path.getsize(os.path.join("data/files", s, f)) for s, f in zip(v.sha, v.filename)]
v["log_size"] = np.log1p(v["size"])
v["agent"] = pd.Categorical(v.agent, categories=[REF] + sorted(set(v.agent) - {REF}))

added_first = v[(v.ver == 0) & (v.origin == "added")].dropna(subset=["repo_id"]).copy()
added_first["y_break"] = added_first.breaking.astype(int)
added_first["y_sec"] = added_first.security_high.astype(int)

last = v[v.ver == v.n_ver - 1]
pr_last = (last.groupby("pr_id")
               .agg(agent=("agent", "first"), repo_id=("repo_id", "first"), merged=("merged", "first"),
                    breaking=("breaking", "any"), security_high=("security_high", "any"),
                    n_ver=("n_ver", "max"))
               .reset_index().dropna(subset=["repo_id"]))
pr_last["y_merged"] = pr_last.merged.astype(int)
pr_last["breaking"] = pr_last.breaking.astype(int)
pr_last["security_high"] = pr_last.security_high.astype(int)
pr_last["log_nver"] = np.log(pr_last.n_ver)
pr_last["agent"] = pd.Categorical(pr_last.agent, categories=list(v.agent.cat.categories))


def gee(formula, data, name):
    data = data.sort_values("repo_id")
    m = smf.gee(formula, groups="repo_id", data=data, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    out = pd.DataFrame({"model": name, "term": m.params.index, "OR": np.exp(m.params),
                        "CI_low": np.exp(ci[0]), "CI_high": np.exp(ci[1]), "p": m.pvalues,
                        "n": int(m.nobs)})
    print(f"\n== {name} (n={int(m.nobs):,}, repos={data.repo_id.nunique():,})")
    print(out.drop(columns=["model", "n"]).round(4).to_string(index=False))
    return out


res = pd.concat([
    gee("y_break ~ agent + log_size", added_first, "M1_breaking_added_first"),
    gee("y_sec ~ agent + log_size", added_first, "M2_security_high_added_first"),
    gee("y_merged ~ breaking + security_high + agent + log_nver", pr_last, "M3_merged"),
])
res.to_csv("results/stats_models.csv", index=False)

rows = []
for y in ["y_break", "y_sec"]:
    tab = pd.crosstab(added_first.agent, added_first[y])
    chi2, p, dof, _ = chi2_contingency(tab)
    cv = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
    rows.append({"outcome": y, "chi2": chi2, "dof": dof, "p": p, "cramers_v": cv})
    print(f"\nchi2 {y}: chi2={chi2:.1f} dof={dof} p={p:.2e} Cramer's V={cv:.3f}")
    print((100 * tab.div(tab.sum(1), axis=0))[1].round(1).rename("pct_positive").to_string())
pd.DataFrame(rows).to_csv("results/stats_chi2.csv", index=False)
