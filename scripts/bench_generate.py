"""Part B — generate workflows with an open model served by vLLM (OpenAI-compatible API), with or without
retrieved action documentation, and repair them in a validator-in-the-loop.

Chains per task (greedy decoding, seed 42):
  zs  : zero-shot prompt -> round 0, then up to K repair rounds driven by validator feedback
  rag : prompt + top-6 retrieved action docs (BM25 over the action catalog) -> round 0, then repair rounds
Conditions (derived at analysis time): C0 = zs round 0, C1 = rag round 0, C2 = zs final, C3 = rag final.
A chain stops early when the validator reports no breaking/security_high/action problems.
Every LLM call is cached in data/bench/gen/<model>/<task>_<chain>_r<k>.json (resumable).
Usage: python scripts/bench_generate.py --model qwen7b --base-url http://localhost:8000/v1 [--limit 10]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from bench_lib import extract_yaml, feedback, load_catalog, render_docs, summarize

K_REPAIR = 3
EXTRA = {}  # extra request fields (--extra-body)
SYSTEM = ("You are an expert DevOps engineer who writes GitHub Actions workflows. "
          "Answer with the complete workflow file in a single ```yaml code block and nothing else.")


def load_context(tid):
    """Repository snapshot before the agent's first commit (bench_fetch_context.py): tree, build files, workflows."""
    p = f"data/bench/context/{tid}.json"
    if not os.path.exists(p):
        return None
    c = json.load(open(p, encoding="utf-8"))
    out = "Repository files (partial tree):\n" + "\n".join(c["tree"]) + "\n"
    for path, text in c["files"].items():
        out += f"\n--- {path} ---\n{text}\n"
    return out


def user_prompt(t, docs=None, context=None):
    p = (f"Repository: {t['repo']} (primary language: {t.get('language') or 'unknown'}).\n"
         f"Create the GitHub Actions workflow file `{t['path']}` that implements the following request.\n\n"
         f"Request:\n{t['request']}\n")
    if context:
        p += f"\nCurrent state of the repository (before your change):\n{context}\n"
    if docs:
        p += f"\nDocumentation of GitHub Actions that may be relevant (use only if appropriate):\n{docs}\n"
    return p + "\nReturn only the complete YAML file."


def call(base, model, messages, max_tokens):
    t0 = time.time()
    for attempt in range(5):
        try:
            r = requests.post(f"{base}/chat/completions", timeout=900, json={
                "model": "m", "messages": messages, "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                **EXTRA})
            if r.status_code == 200:
                d = r.json()
                return {"text": d["choices"][0]["message"].get("content") or "",
                        "finish": d["choices"][0].get("finish_reason"),
                        "prompt_tokens": d["usage"]["prompt_tokens"],
                        "completion_tokens": d["usage"]["completion_tokens"], "latency_s": time.time() - t0}
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return {"text": "", "finish": "error", "prompt_tokens": 0, "completion_tokens": 0, "latency_s": time.time() - t0}


def run_chain(t, chain, a, catalog, bm25):
    out_dir = f"data/bench/gen/{a.model}"
    docs = None
    if chain in ("rag", "ctx"):
        idx = bm25.top(f"{t['request']} {t['path']} {t.get('language') or ''}", k=6)
        docs = render_docs([catalog[i] for i in idx])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user",
                                                       "content": user_prompt(t, docs, load_context(t["task_id"])
                                                                              if chain == "ctx" else None)}]
    for k in range(K_REPAIR + 1):
        dest = f"{out_dir}/{t['task_id']}_{chain}_r{k}.json"
        if os.path.exists(dest):
            rec = json.load(open(dest))
        else:
            res = call(a.base_url, a.model, messages, a.max_tokens)
            y = extract_yaml(res["text"])
            s = summarize(y, t["path"])
            fb = feedback(s)
            rec = {"task_id": t["task_id"], "chain": chain, "round": k, **res, "yaml": y,
                   "eval": {k2: v for k2, v in s.items() if k2 not in ("findings", "action_problems")},
                   "findings": s["findings"], "action_problems": s["action_problems"], "feedback": fb}
            json.dump(rec, open(dest, "w"))
        if not rec["feedback"] or k == K_REPAIR:
            break
        messages += [{"role": "assistant", "content": f"```yaml\n{rec['yaml']}```"},
                     {"role": "user", "content": "Static validators found these problems in your workflow:\n"
                      + "\n".join(rec["feedback"])
                      + "\n\nFix all of them and return the complete corrected workflow in a single ```yaml block."}]
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chains", default="zs,rag", help="comma list of zs, rag, ctx (ctx = rag + repository context)")
    ap.add_argument("--extra-body", default=os.environ.get("EXTRA_BODY", "{}"),
                    help='JSON merged into each request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}')
    a = ap.parse_args()
    EXTRA.update(json.loads(a.extra_body))
    os.makedirs(f"data/bench/gen/{a.model}", exist_ok=True)
    tasks = [json.loads(l) for l in open("data/bench/tasks.jsonl", encoding="utf-8")]
    if a.limit:
        tasks = tasks[: a.limit]
    catalog, bm25 = load_catalog()
    jobs = [(t, c) for t in tasks for c in a.chains.split(",")]
    print(f"model {a.model}: {len(tasks)} tasks, {len(jobs)} chains", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(lambda j: run_chain(j[0], j[1], a, catalog, bm25), jobs), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chains, {time.time() - t0:.0f}s", flush=True)
    print("GEN_DONE", a.model)


if __name__ == "__main__":
    main()
