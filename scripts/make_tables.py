"""Result tables (LaTeX booktabs, one file per table) from the result CSVs. Run from the project root.
Outputs: results/tables/*.tex
"""
import os

import pandas as pd

OUT = "results/tables"
MODEL_LABEL = {"commercial_agent": "Commercial agent (first draft)", "qwen7b": "Qwen2.5-Coder-7B",
               "qwen3coder30b": "Qwen3-Coder-30B-A3B", "gptoss20b": "gpt-oss-20b",
               "gptoss20b_mt4096": "gpt-oss-20b (4k budget)", "qwen36_27b": "Qwen3.6-27B"}
OUT_LABEL = {"breaking": "Breaking", "security_high": "High-sev.\\ security", "supply_chain": "Unpinned",
             "ci_failure": "CI failure"}
DES = {"D1_first_draft": "D1", "D2_final_merged": "D2"}
FINDING_JUDGE = {"gptoss20b": "gpt-oss-20b", "mistral24b": "Mistral-Small-3.2-24B", "qwen3coder30b": "Qwen3-Coder-30B-A3B"}


def write(name, body, cols, caption, label, size="\\small"):
    tex = (f"\\begin{{table}}[t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}"
           f"\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{table}}\n")
    open(f"{OUT}/{name}.tex", "w", encoding="utf-8").write(tex)


def ci(e, lo, hi, d=1):
    return f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def verdict(v, equiv5=False):
    v = v.replace("+-", "$\pm$")
    # a CI that excludes zero can still lie inside the wider margin; say so rather than only "different"
    return v + " (within $\pm$5pp)" if v == "different" and equiv5 else v


def t_rq1():
    d = pd.read_csv("results/v3/power_pooled.csv")
    body = ("Design & Outcome & $n_H$ & $n_A$ & Human \\% & Agent \\% & E1 OR ($p$) & E2 $\\Delta$pp [95\\% CI] & "
            "E3 $\\Delta$pp [95\\% CI] & MDE$_{80}$ & Verdict (E2) \\\\\n\\midrule\n")
    for _, r in d.iterrows():
        body += (f"{DES[r.design]} & {OUT_LABEL[r.outcome]} & {r.n_human:,} & {r.n_agent:,} & {r.rate_human:.1f} & "
                 f"{r.rate_agent:.1f} & {r.E1_OR:.2f} ({r.E1_p:.2f}) & {ci(r.E2_diff_pp, r.E2_lo, r.E2_hi)} & "
                 f"{ci(r.E3_diff_pp, r.E3_lo, r.E3_hi)} & {r.E2_mde80_pp:.1f} & "
                 f"{verdict(r.E2_verdict, r.E2_equiv5)} \\\\\n")
    write("tab_rq1", body, "llrrrrrrrrl",
          "Agent vs.\\ contemporary human workflows (newly added files). E1: GEE logistic; E2: repository fixed "
          "effects LPM; E3: matched on repository $\\times$ size quintile, agent-weighted. All adjust for file size. "
          "MDE$_{80}$: minimum detectable E2 difference (pp) at 80\\% power. Verdict from TOST (90\\% CI) at "
          "$\\pm$3/$\\pm$5\\,pp. CI failure only for versions with a run of the file.", "tab:rq1", "\\scriptsize")


def t_estimand():
    d = pd.read_csv("results/v3/estimand_check.csv")
    d = d[d.estimator.str.contains("globalQ|E2_FE|drop_top10|repo_equal")]
    lab = {"E2_FE": "E2 fixed effects", "E3_ATT_globalQ": "E3 agent-weighted (ATT)",
           "E3_ATC_globalQ": "E3 human-weighted (ATC)", "E3_overlap_globalQ": "E3 overlap weights",
           "E3_equal_globalQ": "E3 equal strata", "E3_ATT_drop_top10_repos": "E3 ATT, top-10 repos dropped",
           "E3_ATT_repo_equal": "E3 ATT, repos equally weighted"}
    body = "Estimator & " + " & ".join(f"{DES[x]} {OUT_LABEL[o]}" for x in DES for o in
                                       ["breaking", "security_high", "supply_chain"]) + " \\\\\n\\midrule\n"
    for e in lab:
        cells = []
        for x in DES:
            for o in ["breaking", "security_high", "supply_chain"]:
                r = d[(d.estimator == e) & (d.design == x) & (d.outcome == o)].iloc[0]
                s = ci(r.est, r.lo, r.hi)
                cells.append(f"\\textbf{{{s}}}" if r.sig else s)
        body += lab[e] + " & " + " & ".join(cells) + " \\\\\n"
    write("tab_estimand", body, "l" + "r" * 6,
          "Sensitivity of the agent$-$human difference (pp, 95\\% repo-cluster bootstrap CI) to the weighting of "
          "repository $\\times$ size strata. Bold: CI excludes zero.", "tab:estimand", "\\tiny")


def t_prevent():
    d = pd.read_csv("results/v2/preventability.csv")
    d = d[(d.target == "failure") & d.signal.isin(["breaking", "unparseable", "security_high", "supply_chain"])]
    body = "Static signal & TP & FP & FN & Precision \\% & Recall \\% \\\\\n\\midrule\n"
    for _, r in d.iterrows():
        body += f"{r.signal.replace('_', ' ')} & {r.tp:,} & {r.fp:,} & {r.fn:,} & {r.precision_pct:.1f} & {r.recall_pct:.1f} \\\\\n"
    write("tab_prevent", body, "lrrrrr",
          "Static findings on the committed YAML as predictors of a failed CI run of the same file at the same "
          "commit (16,317 agent workflow versions with at least one run of that file).", "tab:prevent")


def t_bench():
    r = pd.read_csv("results/bench_v2/bench_rates.csv")
    c = pd.read_csv("results/bench_v2/bench_cost.csv")
    s = pd.read_csv("results/bench_schema.csv").replace({"cond": {"--": "first_draft"}})
    body = ("Model & Cond. & Breaking \\% & Halluc.\\ \\% & Schema-valid \\% & High-sev.\\ \\% & Unpinned \\% & "
            "Calls & Compl.\\ tokens & Latency (s) \\\\\n\\midrule\n")
    prev = None
    for _, x in r.iterrows():
        if prev is not None and x.model != prev:
            body += "\\midrule\n"
        prev = x.model
        cc = c[(c.model == x.model) & (c.cond == x.cond)]
        cost = (f"{cc.calls.iat[0]:.2f} & {cc.completion_tokens.iat[0]:,.0f} & {cc.latency_s.iat[0]:.0f}"
                if len(cc) else "-- & -- & --")
        sv = s[(s.model == x.model) & (s.cond == x.cond)]
        sv = f"{sv.schema_valid.iat[0]:.1f}" if len(sv) else "--"
        body += (f"{MODEL_LABEL[x.model]} & {x.cond.replace('first_draft', '--')} & {x.breaking:.1f} & {x.halluc:.1f} & "
                 f"{sv} & {x.security_high:.1f} & {x.supply_chain:.1f} & {cost} \\\\\n")
    write("tab_bench", body, "llrrrrrrrr",
          "Controlled benchmark on 500 real workflow-authoring tasks. C0 zero-shot; C1 + action-metadata retrieval; "
          "C2 C0 + up to 3 validator-feedback repair rounds; C3 both; C4 C1 + repository snapshot; C5 C4 with repair. "
          "Schema-valid: held-out SchemaStore schema, never used in the loop. Cost columns are means per task.",
          "tab:bench", "\\scriptsize")


def t_judges():
    a = pd.read_csv("results/bench/judge_agreement.csv")
    body = "Pair & $n$ & Agreement & $\\kappa$ \\\\\n\\midrule\n"
    for _, r in a.iterrows():
        agr = "--" if pd.isna(r.agreement) else f"{r.agreement:.2f}"
        if r.pair.startswith("fleiss"):
            pair = "All three (Fleiss)"
        else:
            pair = " vs.\\ ".join(FINDING_JUDGE[j] for j in r.pair.split(" vs "))
        body += f"{pair} & {r.n} & {agr} & {r.kappa:.2f} \\\\\n"
    write("tab_judges", body, "lrrr",
          "Agreement of three blind open-model judges on 120 sampled findings (Cohen's $\\kappa$ per pair, "
          "Fleiss' $\\kappa$ overall).", "tab:judges")


SAMPLE_LABEL = {"S0_all": "All files under any \\texttt{workflows/} path", "S1_active": "Active workflows (main)",
                "S2_no_template": "Active, agent-product templates removed", "S3_dedup": "Active, copies collapsed",
                "S4_strict": "Active, no templates, no copies"}


def t_scope():
    r = pd.read_csv("results/scope_sensitivity.csv")
    outs = ["breaking", "security_high", "supply_chain"]
    body = ("Sample & Design & $n_H$ & $n_A$ & " + " & ".join(f"{OUT_LABEL[o]} $\\Delta$pp [95\\% CI]" for o in outs)
            + " \\\\\n\\midrule\n")
    for s in SAMPLE_LABEL:
        for x in DES:
            g = r[(r["sample"] == s) & (r.design == x)].set_index("outcome")
            cells = []
            for o in outs:
                q = g.loc[o]
                txt = ci(q.E2, q.E2_lo, q.E2_hi)
                cells.append(f"\\textbf{{{txt}}}" if (q.E2_lo > 0 or q.E2_hi < 0) else txt)
            body += (f"{SAMPLE_LABEL[s] if x == 'D1_first_draft' else ''} & {DES[x]} & {int(g.n_human.iat[0]):,} & "
                     f"{int(g.n_agent.iat[0]):,} & " + " & ".join(cells) + " \\\\\n")
    write("tab_scope", body, "llrrrrr",
          "Sensitivity of the within-repository difference (E2) to the sample definition. Bold: CI excludes zero.",
          "tab:scope", "\\scriptsize")


def t_agents():
    r = pd.read_csv("results/scope_sensitivity.csv")
    r = r[r["sample"].str.startswith("A_") & (r.design == "D1_first_draft")]
    body = ("Agent & $n_A$ & " + " & ".join(f"{OUT_LABEL[o]} \\% & $\\Delta$pp (Holm $p$)"
                                           for o in ["breaking", "security_high", "supply_chain"]) + " \\\\\n\\midrule\n")
    for a, g in r.groupby("sample"):
        g = g.set_index("outcome")
        cells = []
        for o in ["breaking", "security_high", "supply_chain"]:
            q = g.loc[o]
            eff = "--" if pd.isna(q.get("E2")) else f"{q.E2:+.1f} ({q.E2_p_holm:.2f})"
            cells.append(f"{q.rate_agent:.1f} & {eff}")
        body += f"{a[2:].replace('_', ' ')} & {int(g.n_agent.iat[0]):,} & " + " & ".join(cells) + " \\\\\n"
    h = r.iloc[0]
    write("tab_agents", body, "lrrrrrrr",
          f"Each agent vs.\\ humans (D1, active workflows without agent-product templates; human rates: breaking "
          f"{r[r.outcome == 'breaking'].rate_human.iat[0]:.1f}\\%, security "
          f"{r[r.outcome == 'security_high'].rate_human.iat[0]:.1f}\\%, unpinned "
          f"{r[r.outcome == 'supply_chain'].rate_human.iat[0]:.1f}\\%). $\\Delta$: E2 within-repository difference; "
          "Holm across agents. Jules has too few shared repositories for E2.", "tab:agents", "\\scriptsize")


def t_security():
    x = pd.read_csv("results/refined_outcomes.csv")
    x = x[(x.design == "D1_first_draft") & (x["sample"] == "active")].set_index("outcome")
    lab = {"exploitable_injection": "Exploitable template injection", "exploitable_pwn": "Exploitable pwn request",
           "exploitable_any": "Any exploitable finding", "unpinned_1p": "Unpinned first-party action",
           "unpinned_3p": "Unpinned third-party action",
           "unpinned_3p_among_users": "Unpinned third-party (files using one)"}
    body = "Outcome & Human \\% & Agent \\% & E1 OR ($p$) & E2 $\\Delta$pp [95\\% CI] & Equivalent within \\\\\n\\midrule\n"
    for o, l in lab.items():
        q = x.loc[o]
        if o.startswith("exploitable"):
            eq = "$\\pm$1\\,pp" if q.tost1_p < .05 else ("$\\pm$2\\,pp" if q.tost2_p < .05 else "--")
        else:
            eq = "$\\pm$3\\,pp" if q.tost3_p < .05 else ("$\\pm$5\\,pp" if q.tost5_p < .05 else "--")
        body += (f"{l} & {q.rate_human:.1f} & {q.rate_agent:.1f} & {q.E1_OR:.2f} ({q.E1_p:.2f}) & "
                 f"{ci(q.E2, q.E2_lo, q.E2_hi, 2 if o.startswith('exploitable') else 1)} & {eq} \\\\\n")
    write("tab_security", body, "lrrrrl",
          "Refined security outcomes (D1, active workflows). Exploitable: outsider-reachable trigger and "
          "attacker-controlled input in a run block (injection) or untrusted checkout in a privileged trigger "
          "(pwn request).", "tab:security", "\\scriptsize")

CAND_LABEL = {"reference": "Landed final version (sanity check)", "qwen36_27b:C3": "Qwen3.6-27B, C3",
              "qwen3coder30b:C3": "Qwen3-Coder-30B-A3B, C3", "qwen3coder30b:C5": "Qwen3-Coder-30B-A3B, C5",
              "qwen3coder30b:C0": "Qwen3-Coder-30B-A3B, C0", "gptoss20b:C3": "gpt-oss-20b, C3",
              "qwen7b:C2": "Qwen2.5-Coder-7B, C2"}
JUDGE_LABEL = {"mistral24b": "Mistral", "gemma27b": "Gemma", "consensus": "Consensus"}


def t_pairwise():
    x = pd.read_csv("results/bench_pairwise.csv")
    judges = [j for j in JUDGE_LABEL if j in set(x.judge)]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{JUDGE_LABEL[j]}}}" for j in judges)
    sub = " & ".join("W/T/L & Net" for _ in judges)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(judges)))
    body = f"Candidate vs.\\ agent's first draft & {head} \\\\\n{rules}\n & {sub} \\\\\n\\midrule\n"
    for c, l in CAND_LABEL.items():
        cells = []
        for j in judges:
            q = x[(x.judge == j) & (x.candidate == c)].iloc[0]
            star = "$^{*}$" if q.sign_p_holm < .05 else ""
            cells.append(f"{q.win:.0f}/{q.tie:.0f}/{q.loss:.0f} & {q.net:+.0f}{star}")
        body += f"{l} & {' & '.join(cells)} \\\\\n"
    write("tab_pairwise", body, "l" + "rr" * len(judges),
          "Blind pairwise task fulfilment against the commercial agent's first draft (500 tasks). W/T/L: \\% of "
          "tasks the candidate wins, ties or loses; a win counts only if it holds in both presentation orders. "
          "Net = W$-$L in pp; $^{*}$ exact sign test, Holm-adjusted $p<.05$ within judge. Consensus: both judges "
          "agree, else tie.", "tab:pairwise", "\\scriptsize")


def t_human():
    f = pd.read_csv("results/human_validation_findings.csv")
    c = pd.read_csv("results/human_precision_corrected.csv")
    c = c[(c.rater == "consensus") & (c.variant == "unsure_excluded")].set_index(["category", "group"])
    pr = pd.read_csv("results/human_validation_pairs.csv").set_index(["rater", "candidate"])
    ag = pd.read_csv("results/human_validation_agreement.csv")
    k_f = ag[(ag.part == "findings") & (ag.a == "R1") & (ag.b == "R2")].kappa.iloc[0]
    k_p = ag[(ag.part == "pairs") & (ag.a == "R1") & (ag.b == "R2")].kappa.iloc[0]
    E = " \\\\\n"  # LaTeX row end
    body = ("\\multicolumn{6}{l}{\\emph{(a) Precision of static findings (120 findings, author-blinded)}}" + E +
            "Category & Author & R1 & R2 & Consensus [95\\% CI] & Confirmed-problem rate \\% [95\\% CI]" + E +
            "\\midrule\n")
    for cat, lab in (("breaking", "Breaking"), ("security_high", "High-sev.\\ security")):
        for grp in ("agent", "human"):
            g = lambda r: f[(f.rater == r) & (f.category == cat) & (f.group == grp)].iloc[0]
            q = g("consensus")
            cc = c.loc[(cat, grp)]
            body += (f"{lab} & {grp.capitalize()} & {g('R1').precision:.2f} & {g('R2').precision:.2f} & "
                     f"{q.precision:.2f} [{q.lo:.2f}, {q.hi:.2f}] & {cc.corrected_rate:.1f} [{cc.lo:.1f}, {cc.hi:.1f}]" + E)
        d = c.loc[(cat, "agent-human")]
        body += f"\\multicolumn{{5}}{{r}}{{Agent $-$ human}} & {d.corrected_rate:+.1f} [{d.lo:+.1f}, {d.hi:+.1f}]" + E
    body += ("\\midrule\n\\multicolumn{6}{l}{\\emph{(b) Task fulfilment vs.\\ the agent's first draft (40 tasks): "
             "W/T/L \\% and net}}" + E + "Candidate & & R1 & R2 & Consensus & Mistral / Gemma (same tasks)" + E +
             "\\midrule\n")
    wtl = lambda r, cd: f"{pr.loc[(r, cd)].win:.0f}/{pr.loc[(r, cd)].tie:.0f}/{pr.loc[(r, cd)].loss:.0f}"
    for cd, lab in (("qwen36_27b:C3", "Qwen3.6-27B, C3"), ("qwen3coder30b:C3", "Qwen3-Coder-30B-A3B, C3"), ("all", "Both")):
        star = "$^{*}$" if pr.loc[("consensus", cd)].sign_p < .05 else ""
        body += (f"{lab} & & {pr.loc[('R1', cd)].net:+.0f} & {pr.loc[('R2', cd)].net:+.0f} & "
                 f"{wtl('consensus', cd)} ({pr.loc[('consensus', cd)].net:+.0f}){star} & "
                 f"{pr.loc[('llm:mistral24b', cd)].net:+.0f} / {pr.loc[('llm:gemma27b', cd)].net:+.0f}" + E)
    # second round: Qwen3-Coder-30B-A3B with repository context (C5), same 20 tasks + 40 new ones
    cx = pd.read_csv("results/human_ctx_pairs.csv").set_index(["rater", "part"])
    pc = pd.read_csv("results/human_ctx_paired_c3.csv").set_index("rater").loc["consensus"]
    k_c = pd.read_csv("results/human_ctx_agreement.csv").query("a == 'R1' and b == 'R2' and part == 'all'").kappa.iloc[0]
    body += ("\\midrule\n\\multicolumn{6}{l}{\\emph{(c) Second round: Qwen3-Coder-30B-A3B with repository context (C5) "
             "vs.\\ the agent's first draft}}" + E + "\\midrule\n")
    h = lambda x, s="": f"{x:{s}.1f}".replace(".0", "")  # 40-task shares end in .5; keep it instead of rounding
    for part, lab in (("same", "C5, tasks of round (b)"), ("new", "C5, new tasks"), ("all", "C5, all")):
        q = cx.loc[("consensus", part)]
        star = "$^{*}$" if q.sign_p < .05 else ""
        body += (f"{lab} ({q.n:.0f}) & & {h(cx.loc[('R1', part)].net, '+')} & {h(cx.loc[('R2', part)].net, '+')} & "
                 f"{h(q.win)}/{h(q.tie)}/{h(q.loss)} ({h(q.net, '+')}){star} & "
                 f"{h(cx.loc[('llm:mistral24b', part)].net, '+')} / {h(cx.loc[('llm:gemma27b', part)].net, '+')}" + E)
    write("tab_human", body, "llrrll",
          f"Two-rater human validation (R1: senior DevOps engineer; R2: platform engineer). (a) Share of findings judged true problems (30 findings per category and "
          f"author group; UNSURE excluded); consensus requires both raters to agree. Confirmed-problem rate = D1 flag rate $\\times$ consensus precision "
          f"(bootstrap CI over annotated items). (b) Net = W$-$L in pp; $^{{*}}$ exact sign test $p<.05$. "
          f"Inter-rater Cohen's $\\kappa$: {k_f:.2f} (findings), {k_p:.2f} (task fulfilment), {k_c:.2f} (second round). "
          f"(c) On the 20 tasks of round (b), the consensus verdict moved from C3 to C5 in the model's favour on "
          f"{pc.better:.0f} tasks and against it on {pc.worse:.0f} (sign test $p={pc.sign_p:.2f}$).", "tab:human",
          "\\scriptsize")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (t_rq1, t_estimand, t_prevent, t_bench, t_judges, t_scope, t_agents, t_security, t_pairwise, t_human):
        f()
        print("ok", f.__name__)
