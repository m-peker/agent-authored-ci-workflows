"""Run static validators on every fetched workflow file version.

  * actionlint 1.7.12 (workflows only; shellcheck/pyflakes integrations disabled for determinism)
  * zizmor 1.31.1 --offline (security audits; workflows + composite actions)

Output: data/processed/findings.parquet with columns
  sha, filename, tool, rule, severity, line, message
plus data/processed/linted_files.parquet (every file that was linted, so files with zero findings count too).
Resumable per chunk: data/interim/lint/<tool>_<chunk>.parquet
Usage: python scripts/run_linters.py [--limit N] [--procs P] [--root data/files_human --tag _human]
"""
import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = "data/files"
OUT = "data/interim/lint"
ACTIONLINT = "tools/actionlint"
ZIZMOR = ".venv/bin/zizmor"
CHUNK = 200


def key_of(path: str):
    rel = os.path.relpath(path, ROOT)
    sha, filename = rel.split(os.sep, 1)
    return sha, filename.replace(os.sep, "/")


def run_actionlint(paths):
    wf = [p for p in paths if "/.github/workflows/" in p.replace(os.sep, "/")]
    if not wf:
        return []
    res = subprocess.run([ACTIONLINT, "-format", "{{json .}}", "-shellcheck=", "-pyflakes=", *wf],
                         capture_output=True, text=True, timeout=600)
    rows = []
    for e in json.loads(res.stdout or "[]"):
        sha, fn = key_of(e["filepath"])
        rows.append((sha, fn, "actionlint", e.get("kind"), "error", e.get("line"), e.get("message")))
    return rows


def _zizmor_path(f):
    for loc in f.get("locations", []):
        key = loc.get("symbolic", {}).get("key", {})
        local = key.get("Local") or key.get("local") or {}
        p = local.get("given_path") or local.get("verbatim_path") or local.get("path")
        if p:
            line = loc.get("concrete", {}).get("location", {}).get("start_point", {}).get("row")
            return p, (line + 1 if isinstance(line, int) else None)
    return None, None


ZIZMOR_OK = {0, 10, 11, 12, 13, 14}  # 0 = clean, 10-14 = findings by max severity


def run_zizmor(paths):
    res = subprocess.run([ZIZMOR, "--offline", "--format", "json", "--no-progress", *paths],
                         capture_output=True, text=True, timeout=900)
    if res.returncode not in ZIZMOR_OK:
        # One unparseable file aborts the whole batch: fall back to one file at a time.
        if len(paths) > 1:
            return [r for p in paths for r in run_zizmor([p])]
        sha, fn = key_of(paths[0])
        return [(sha, fn, "zizmor", "_input_error", None, None, res.stderr[-300:])]
    rows = []
    try:
        findings = json.loads(res.stdout or "[]")
    except json.JSONDecodeError:
        return [("_error", "_error", "zizmor", "parse_error", None, None, res.stderr[-500:])]
    for f in findings:
        p, line = _zizmor_path(f)
        if not p:
            continue
        sha, fn = key_of(p)
        det = f.get("determinations", {})
        # severity stored as "<severity>/<confidence>", e.g. "High/High"
        sev = f"{det.get('severity')}/{det.get('confidence')}"
        rows.append((sha, fn, "zizmor", f.get("ident"), sev, line, f.get("desc")))
    return rows


def work(args):
    idx, paths, root, outdir = args
    global ROOT
    ROOT = root  # worker processes need the root for key_of()
    out = f"{outdir}/chunk_{idx:05d}.parquet"
    if os.path.exists(out):
        return idx
    rows = run_actionlint(paths) + run_zizmor(paths)
    pd.DataFrame(rows, columns=["sha", "filename", "tool", "rule", "severity", "line", "message"]) \
      .to_parquet(out, index=False)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--root", default="data/files")
    ap.add_argument("--tag", default="", help="suffix for interim dir and outputs, e.g. _human")
    a = ap.parse_args()
    global ROOT
    ROOT, outdir = a.root, OUT + a.tag
    os.makedirs(outdir, exist_ok=True)
    # glob skips dot-directories such as .github, so walk the tree instead
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs
                   if f.endswith((".yml", ".yaml")))
    if a.limit:
        files = files[: a.limit]
    print(f"files: {len(files):,}", flush=True)
    chunks = [(i, files[s:s + CHUNK], ROOT, outdir) for i, s in enumerate(range(0, len(files), CHUNK))]
    with ProcessPoolExecutor(a.procs) as ex:
        for n, _ in enumerate(ex.map(work, chunks), 1):
            if n % 50 == 0 or n == len(chunks):
                print(f"chunks {n}/{len(chunks)}", flush=True)
    pd.DataFrame([key_of(p) for p in files], columns=["sha", "filename"]) \
      .to_parquet(f"data/processed/linted_files{a.tag}.parquet", index=False)
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{outdir}/chunk_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(f"data/processed/findings{a.tag}.parquet", index=False)
    print("DONE findings:", len(df))
    print(df.groupby(["tool"]).size().to_string())
    print(df.groupby(["tool", "rule"]).size().sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
