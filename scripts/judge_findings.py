"""Blind LLM adjudication of the 120 sampled static findings (construct validity of `breaking`/`security_high`).

Each judge (open model served by vLLM) sees the finding (tool, rule, message, line) and the FULL workflow file
— never the author (agent vs human). It answers TP / FP / UNSURE with a short reason, using the same criteria as
the human annotation guide. Greedy decoding, seed 42; cached per (judge, item).
Output: data/annotation/judge_<judge>.jsonl
Usage: python scripts/judge_findings.py --judge qwen3coder30b [--base-url http://localhost:8000/v1] [--limit 5]
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

CRITERIA = {
    "breaking": ("TP if the reported problem would make the workflow fail, behave incorrectly, or be rejected by "
                 "GitHub (e.g. undefined step output or property, wrong `needs`, YAML/syntax error, invalid "
                 "permission or event, wrong expression type). FP if it is only style, an outdated version, a "
                 "linter limitation, or the construct is actually valid on GitHub Actions."),
    "security_high": ("TP if a realistic attacker could exploit it (e.g. attacker-controlled PR/issue text "
                      "expanded inside `run:`, `pull_request_target` that checks out and runs PR code, write "
                      "permissions or secrets exposed to untrusted code). FP if the input is trusted (e.g. "
                      "github.sha, maintainer-only workflow_dispatch input), the trigger cannot be reached by "
                      "outsiders, or the risk is purely theoretical."),
}
SYSTEM = ("You are a senior GitHub Actions and CI/CD security reviewer. You judge whether a static-analysis "
          "finding is a true problem. Be strict and concrete. Answer ONLY with a JSON object: "
          '{"label": "TP" | "FP" | "UNSURE", "impact": "HIGH" | "MEDIUM" | "LOW" | null, "reason": "<one sentence>"}')
JSON_RE = re.compile(r"\{.*\}", re.S)


def prompt(it, text):
    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1:4d} | {l}" for i, l in enumerate(lines[:400]))
    return (f"Finding category: {it['category']}\nTool/rule: {it['tool']}:{it['rule']}\n"
            f"Reported line: {it['line']}\nMessage: {it['message']}\n\n"
            f"Labelling rule: {CRITERIA[it['category']]}\n\n"
            f"Workflow file (line-numbered):\n```yaml\n{numbered}\n```\n\nReturn the JSON object only.")


def judge(it, text, a):
    dest = f"data/annotation/judge_cache/{a.judge}_{it['item_id']}.json"
    if os.path.exists(dest):
        return json.load(open(dest))
    out = {"item_id": it["item_id"], "judge": a.judge, "label": None, "impact": None, "reason": None, "raw": ""}
    for attempt in range(3):
        try:
            r = requests.post(f"{a.base_url}/chat/completions", timeout=900, json={
                "model": "m", "temperature": 0, "seed": 42, "max_tokens": a.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(it, text)}]})
            txt = r.json()["choices"][0]["message"].get("content") or ""
            out["raw"] = txt[-2000:]
            m = JSON_RE.search(txt)
            if m:
                j = json.loads(m.group(0))
                lab = str(j.get("label", "")).upper()
                out.update(label=lab if lab in ("TP", "FP", "UNSURE") else None, impact=j.get("impact"),
                           reason=j.get("reason"))
            if out["label"]:
                break
        except (requests.RequestException, ValueError, KeyError):
            continue
    json.dump(out, open(dest, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=6000)
    a = ap.parse_args()
    os.makedirs("data/annotation/judge_cache", exist_ok=True)
    items = [json.loads(l) for l in open("data/annotation/sample.jsonl", encoding="utf-8")]
    key = pd.read_csv("data/annotation/key.csv").set_index("item_id")  # only used to locate the file
    if a.limit:
        items = items[: a.limit]

    def run(it):
        k = key.loc[it["item_id"]]
        root = "data/files" if k.group == "agent" else "data/files_human"
        text = open(os.path.join(root, k.sha, k.filename), encoding="utf-8", errors="replace").read()
        return judge(it, text, a)

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, items))
    with open(f"data/annotation/judge_{a.judge}.jsonl", "w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(a.judge, pd.Series([r["label"] for r in res]).value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
