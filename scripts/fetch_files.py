"""Fetch full workflow file contents at each PR commit from raw.githubusercontent.com (no token needed).

Cache layout: data/files/<sha>/<path>; failures logged to data/files/_failed.tsv. Resumable.
Usage: python scripts/fetch_files.py --input data/processed/human_wf_changes.parquet --root data/files_human
       python scripts/fetch_files.py --sample 200      (stratified by agent, seed 42; smoke test)
       python scripts/fetch_files.py                   (all non-removed changes)
"""
import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = "data/files"
FAILED = f"{ROOT}/_failed.tsv"  # reset in main() when --root is given
lock = threading.Lock()
session = requests.Session()
session.headers["User-Agent"] = "agentci-research (academic study of CI workflows)"


def repo_full_name(api_url: str) -> str:
    return api_url.replace("https://api.github.com/repos/", "")


def fetch(row) -> str:
    dest = os.path.join(ROOT, row.sha, row.filename)
    if os.path.exists(dest):
        return "cached"
    url = f"https://raw.githubusercontent.com/{repo_full_name(row.repo)}/{row.sha}/{row.filename}"
    r = None
    for attempt in range(6):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(120, 5 * 2 ** attempt))
            continue
        if r.status_code == 200:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(r.content)
            return "ok"
        if r.status_code in (404, 451):
            break
    with lock, open(FAILED, "a", encoding="utf-8") as fh:
        fh.write(f"{row.sha}\t{row.filename}\t{getattr(r, 'status_code', 'exc')}\n")
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--input", default="data/processed/wf_changes_clean.parquet")
    ap.add_argument("--root", default="data/files")
    args = ap.parse_args()
    global ROOT, FAILED
    ROOT, FAILED = args.root, f"{args.root}/_failed.tsv"
    os.makedirs(ROOT, exist_ok=True)

    wf = pd.read_parquet(args.input)
    if "repo" not in wf.columns:  # agent table has repo_url, human table has repo
        wf["repo"] = wf.repo_url
    wf = wf[wf.status != "removed"].drop_duplicates(["sha", "filename"])
    if args.sample:
        per = max(1, args.sample // wf.agent.nunique())
        wf = (wf.groupby("agent", group_keys=False)
                .apply(lambda g: g.sample(min(per, len(g)), random_state=42)))
    print(f"to fetch: {len(wf):,}", flush=True)

    stats = {"ok": 0, "cached": 0, "fail": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, r) for r in wf.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            stats[f.result()] += 1
            if i % 2000 == 0 or i == len(futs):
                print(f"{i:,}/{len(futs):,} {stats}", flush=True)
    print("DONE", stats)


if __name__ == "__main__":
    random.seed(42)
    main()
