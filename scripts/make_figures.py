"""Figures (PDF) from the result CSVs. Run from the project root: python scripts/make_figures.py
F1 fig_rq1_forest.pdf     agent - human difference (pp) per design x outcome, E2 within-repo and E3 matched, 95% CI,
                          with the +-3 / +-5 pp equivalence margins.
F2 fig_bench_conditions.pdf  breaking and hallucination rate per open model x condition, agent first draft as reference.
F3 fig_repair_budget.pdf  breaking rate vs mean LLM calls for repair budget k = 0..3 (zs and rag chains).
F4 fig_judges.pdf         consensus-free precision of static findings per LLM judge and majority vote.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-4, fixed order
MODEL_ORDER = ["qwen7b", "qwen3coder30b", "qwen36_27b", "gptoss20b"]
MODEL_LABEL = {"qwen7b": "Qwen2.5-Coder-7B", "qwen3coder30b": "Qwen3-Coder-30B-A3B", "gptoss20b": "gpt-oss-20b",
               "gptoss20b_mt4096": "gpt-oss-20b (4k budget)", "qwen36_27b": "Qwen3.6-27B"}
MODEL_COLOR = {"qwen7b": S[0], "qwen3coder30b": S[1], "gptoss20b": S[2], "gptoss20b_mt4096": S[2], "qwen36_27b": S[3]}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})


def fig_forest():
    d = pd.read_csv("results/v3/power_pooled.csv")
    outs = ["breaking", "security_high", "supply_chain", "ci_failure"]
    lab = {"breaking": "Breaking", "security_high": "High-severity security", "supply_chain": "Unpinned (supply chain)",
           "ci_failure": "CI run failure"}
    des = {"D1_first_draft": "D1 first draft", "D2_final_merged": "D2 merged final"}
    rows = [(o, dz) for o in outs for dz in des]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for i, (o, dz) in enumerate(rows):
        r = d[(d.outcome == o) & (d.design == dz)].iloc[0]
        y = len(rows) - 1 - i
        for off, est, c, name in ((0.15, "E2", S[0], "Within-repo FE (E2)"), (-0.15, "E3", S[1], "Matched repo x size (E3)")):
            ax.errorbar(r[f"{est}_diff_pp"], y + off, xerr=[[r[f"{est}_diff_pp"] - r[f"{est}_lo"]],
                        [r[f"{est}_hi"] - r[f"{est}_diff_pp"]]], fmt="o", ms=4.5, color=c, elinewidth=1.6, capsize=0,
                        label=name if i == 0 else None)
    for m, a in ((3, 0.10), (5, 0.05)):
        ax.axvspan(-m, m, color=INK2, alpha=a, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab[o]} · {des[dz]}" for o, dz in rows][::-1])
    ax.set_xlabel("Agent − human difference (pp), 95% CI")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_rq1_forest.pdf", bbox_inches="tight")


def fig_conditions():
    r = pd.read_csv("results/bench_v2/bench_rates.csv")
    agent = r[r.model == "commercial_agent"].iloc[0]
    conds = ["C0", "C1", "C2", "C3"]
    clab = ["C0\nzero-shot", "C1\n+RAG", "C2\n+repair", "C3\nRAG+repair"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=False)
    for ax, o, t in zip(axes, ["breaking", "halluc"], ["Breaking workflows (%)", "Hallucinated action/input (%)"]):
        x = np.arange(len(conds))
        w = 0.2
        for j, m in enumerate(MODEL_ORDER):
            g = r[r.model == m].set_index("cond").loc[conds]
            ax.bar(x + (j - 1.5) * w, g[o], w - 0.03, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.axhline(agent[o], color=INK, lw=1, ls="--", label=f"Commercial agent, first draft")
        ax.set_xticks(x)
        ax.set_xticklabels(clab, fontsize=7.5)
        ax.set_title(t, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_bench_conditions.pdf", bbox_inches="tight")


def fig_repair():
    d = pd.read_csv("results/bench_v2/bench_ablate_k.csv")
    agent = pd.read_csv("results/bench_v2/bench_rates.csv").query("model == 'commercial_agent'").breaking.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, ch, t in zip(axes, ["zs", "rag"], ["Zero-shot + repair", "RAG + repair"]):
        for m in MODEL_ORDER + ["gptoss20b_mt4096"]:
            g = d[(d.model == m) & (d.chain == ch)].sort_values("k")
            ls = ":" if m == "gptoss20b_mt4096" else "-"
            ax.plot(g.calls, g.breaking, ls, marker="o", ms=4, lw=1.6, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            for _, p in g.iterrows():
                if m == "qwen7b" or p.k in (0, 3):
                    ax.annotate(f"k={int(p.k)}", (p.calls, p.breaking), textcoords="offset points", xytext=(3, 3),
                                fontsize=6, color=INK2)
        ax.axhline(agent, color=INK, lw=1, ls="--", label=f"Commercial agent ({agent:.1f}%)")
        ax.set_ylim(0, 35)
        ax.set_title(t, fontsize=8.5, loc="left", color=INK)
        ax.set_xlabel("Mean LLM calls per task")
    axes[0].set_ylabel("Breaking workflows (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_repair_budget.pdf", bbox_inches="tight")


def fig_judges():
    d = pd.read_csv("results/bench/judged_items.csv")
    judges = ["qwen3coder30b", "gptoss20b", "mistral24b", "consensus"]
    jl = {"qwen3coder30b": "Qwen3-Coder-30B", "gptoss20b": "gpt-oss-20b", "mistral24b": "Mistral-Small-24B",
          "consensus": "Majority (≥2/3)"}
    from statsmodels.stats.proportion import proportion_confint
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for k, (cat, c) in enumerate((("breaking", S[0]), ("security_high", S[1]))):
        g = d[d.category == cat]
        for i, j in enumerate(judges):
            tp, fp = (g[j] == "TP").sum(), (g[j] == "FP").sum()
            p = tp / (tp + fp)
            lo, hi = proportion_confint(tp, tp + fp, method="wilson")
            y = i + (0.17 if k == 0 else -0.17)
            ax.errorbar(100 * p, y, xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="o", ms=4.5, color=c,
                        elinewidth=1.6, label=("Breaking" if k == 0 else "High-severity security") if i == 0 else None)
    ax.set_yticks(range(len(judges)))
    ax.set_yticklabels([jl[j] for j in judges])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Judged precision TP/(TP+FP) (%), Wilson 95% CI")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_judges.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_forest, fig_conditions, fig_repair, fig_judges):
        f()
        print("ok", f.__name__)
