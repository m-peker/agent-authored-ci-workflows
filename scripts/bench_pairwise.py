"""Task fulfilment, independent of the validators: blind pairwise comparison against the commercial agent's draft.

For each task and candidate (model x condition final output), a judge model sees the request, the repository name
and the two workflows as A and B (no authorship, no validator output) and answers which better fulfils the request
(A, B or TIE). Each pair is judged twice with positions swapped; a side wins only if it wins in both orders,
otherwise TIE (controls position bias). Sanity check: the landed reference vs the agent draft.
Judges must not belong to a contestant model family. Greedy, seed 42; cached per (judge, pair, order).
Output: data/bench/pairwise/<judge>.jsonl ; summary via bench_pairwise_summary in this file (--summary).
Usage: python scripts/bench_pairwise.py --judge mistral24b --candidates qwen3coder30b:C3,qwen3coder30b:C5,...
"""
import argparse
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

SYSTEM = ("You are a senior DevOps engineer reviewing two candidate GitHub Actions workflows written for the same "
          "request. Judge ONLY how well each workflow fulfils the request as a working CI/CD workflow for this "
          "repository: correct triggers, the jobs and steps the request asks for, sensible use of actions, and "
          "whether it would plausibly work. Ignore formatting, comments and length. Answer with a JSON object: "
          '{"winner": "A" | "B" | "TIE", "reason": "<one sentence>"}')
COND = {("zs", "first"): "C0", ("rag", "first"): "C1", ("zs", "final"): "C2", ("rag", "final"): "C3",
        ("ctx", "first"): "C4", ("ctx", "final"): "C5"}
JSON_RE = re.compile(r"\{.*\}", re.S)


def candidate_yaml(model, cond):
    out = {}
    recs = {}
    for f in glob.glob(f"data/bench/gen/{model}/*.json"):
        r = json.load(open(f))
        recs.setdefault((r["task_id"], r["chain"]), []).append(r)
    for (tid, chain), rs in recs.items():
        rs.sort(key=lambda r: int(r["round"]))
        for which, pick in (("first", rs[0]), ("final", rs[-1])):
            if COND[(chain, which)] == cond:
                out[tid] = pick["yaml"]
    return out


def ask(a, t, ya, yb, key):
    dest = f"data/bench/pairwise/cache/{a.judge}/{key}.json"
    if os.path.exists(dest):
        return json.load(open(dest))
    user = (f"Repository: {t['repo']} (language: {t.get('language') or 'unknown'}), target file `{t['path']}`.\n\n"
            f"Request:\n{t['request']}\n\n--- Workflow A ---\n```yaml\n{ya[:12000]}\n```\n\n--- Workflow B ---\n"
            f"```yaml\n{yb[:12000]}\n```\n\nWhich workflow better fulfils the request? Return the JSON object only.")
    res = {"winner": None, "reason": None}
    for _ in range(3):
        try:
            r = requests.post(f"{a.base_url}/chat/completions", timeout=900, json={
                "model": "m", "temperature": 0, "seed": 42, "max_tokens": a.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
            txt = r.json()["choices"][0]["message"].get("content") or ""
            m = JSON_RE.search(txt)
            if m:
                j = json.loads(m.group(0))
                w = str(j.get("winner", "")).upper()
                if w in ("A", "B", "TIE"):
                    res = {"winner": w, "reason": j.get("reason")}
                    break
        except (requests.RequestException, ValueError, KeyError):
            continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(res, open(dest, "w"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--candidates", required=True, help="comma list model:cond, plus 'reference'")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")][: a.limit]
    agent = {}
    ref = {}
    for t in tasks:
        for store, key in ((agent, "first_sha"), (ref, "ref_sha")):
            p = os.path.join("data/files", t[key], t["path"])
            if os.path.exists(p):
                store[t["task_id"]] = open(p, encoding="utf-8", errors="replace").read()
    jobs = []
    for c in a.candidates.split(","):
        cand = ref if c == "reference" else candidate_yaml(*c.split(":"))
        for t in tasks:
            tid = t["task_id"]
            if tid in cand and tid in agent:
                jobs.append((c, t, cand[tid], agent[tid]))

    def run(j):
        c, t, y_c, y_a = j
        tag = c.replace(":", "_")
        r1 = ask(a, t, y_c, y_a, f"{tag}_{t['task_id']}_cand_first")   # candidate = A
        r2 = ask(a, t, y_a, y_c, f"{tag}_{t['task_id']}_agent_first")  # candidate = B
        c1 = {"A": "cand", "B": "agent", "TIE": "tie"}.get(r1["winner"])
        c2 = {"A": "agent", "B": "cand", "TIE": "tie"}.get(r2["winner"])
        final = c1 if c1 == c2 and c1 in ("cand", "agent") else "tie"
        return {"candidate": c, "task_id": t["task_id"], "order1": c1, "order2": c2, "result": final,
                "consistent": c1 == c2}

    os.makedirs("data/bench/pairwise", exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run, jobs))
    with open(f"data/bench/pairwise/{a.judge}.jsonl", "w", encoding="utf-8") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    import pandas as pd
    d = pd.DataFrame(res)
    print(d.groupby("candidate").result.value_counts(normalize=True).unstack().round(3).to_string())
    print("position-consistent share:", round(d.consistent.mean(), 3))


if __name__ == "__main__":
    main()
