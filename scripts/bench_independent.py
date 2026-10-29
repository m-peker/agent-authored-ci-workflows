"""Validator-independent check of benchmark outputs (the repair loop optimises actionlint/zizmor).

Held-out validator: the SchemaStore JSON schema for GitHub workflows (tools/schemastore/github-workflow.json,
draft-07), never shown to the models and not used in the repair loop. A workflow is schema-valid if it parses as
YAML and validates against the schema. PyYAML's `on: -> True` key quirk is undone before validation.
Evaluated: every model x condition (C0..C5 where present), the commercial agent's first draft and the landed
reference. Paired McNemar tests of each condition vs the agent draft (Holm within model).
Output: results/bench_schema.csv, results/bench_schema_tests.csv
"""
import glob
import json
import os

import jsonschema
import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from bench_analyze import boot, mcnemar

SCHEMA = json.load(open("tools/schemastore/github-workflow.json", encoding="utf-8"))
VALIDATOR = jsonschema.Draft7Validator(SCHEMA)
COND = {("zs", "first"): "C0", ("rag", "first"): "C1", ("zs", "final"): "C2", ("rag", "final"): "C3",
        ("ctx", "first"): "C4", ("ctx", "final"): "C5"}


class Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans (only true/false), as GitHub parses workflows: `on`, `yes`, `no` stay strings."""


Yaml12Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
Yaml12Loader.add_implicit_resolver("tag:yaml.org,2002:bool", __import__("re").compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
                                   list("tTfF"))


def valid(text):
    try:
        d = yaml.load(text, Loader=Yaml12Loader)
    except yaml.YAMLError:
        return False
    if not isinstance(d, dict):
        return False
    try:
        return VALIDATOR.is_valid(d)
    except TypeError:  # non-string mapping keys (e.g. numbers) -> not a valid workflow mapping
        return False


def main():
    tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")]
    agent = pd.read_json("data/bench/agent_eval.jsonl", lines=True) if os.path.exists("data/bench/agent_eval.jsonl") else None
    rows = []
    for m in sorted(os.listdir("data/bench/gen")):
        recs = {}
        for f in glob.glob(f"data/bench/gen/{m}/*.json"):
            r = json.load(open(f))
            recs.setdefault((r["task_id"], r["chain"]), []).append(r)
        for (tid, chain), rs in recs.items():
            rs.sort(key=lambda r: int(r["round"]))
            for which, pick in (("first", rs[0]), ("final", rs[-1])):
                rows.append({"model": m, "cond": COND[(chain, which)], "task_id": tid, "schema_valid": valid(pick["yaml"])})
    # agent first draft and landed reference (texts stored by bench_eval_agents.py, else read from data/files)
    for t in tasks:
        for who, key in (("commercial_agent", "first_sha"), ("reference", "ref_sha")):
            p = os.path.join("data/files", t[key], t["path"])
            if os.path.exists(p):
                rows.append({"model": who, "cond": "--", "task_id": t["task_id"],
                             "schema_valid": valid(open(p, encoding="utf-8", errors="replace").read())})
    d = pd.DataFrame(rows)
    rates = []
    for (m, c), g in d.groupby(["model", "cond"]):
        est, lo, hi = boot(g.schema_valid)
        rates.append({"model": m, "cond": c, "n": len(g), "schema_valid": est, "lo": lo, "hi": hi})
    rates = pd.DataFrame(rates).round(1)
    rates.to_csv("results/bench_schema.csv", index=False)
    print(rates.to_string(index=False))
    ag = d[d.model == "commercial_agent"].set_index("task_id").schema_valid
    tests = []
    for m, g in d[~d.model.isin(["commercial_agent", "reference"])].groupby("model"):
        block = []
        for c, gc in g.groupby("cond"):
            x = gc.set_index("task_id").schema_valid
            common = x.index.intersection(ag.index)
            n01, n10, p = mcnemar(x[common].values, ag[common].values)
            block.append({"model": m, "comparison": f"{c} vs agent", "only_model": n10, "only_agent": n01, "p": p})
        for b, ph in zip(block, multipletests([b["p"] for b in block], method="holm")[1]):
            b["p_holm"] = ph
        tests += block
    t = pd.DataFrame(tests).round(4)
    t.to_csv("results/bench_schema_tests.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
