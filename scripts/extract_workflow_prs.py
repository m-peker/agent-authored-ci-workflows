"""Find AIDev PR commits that touch GitHub Actions workflow / action definition files.

Reads only light columns of the remote pr_commit_details.parquet (no patch text), row group by row group,
so it is resumable: each row group's matches are saved to data/interim/wf_rg/<g>.parquet.

Usage: python scripts/extract_workflow_prs.py [--max-groups N]   (N for smoke tests)
"""
import argparse
import os
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SRC = "datasets/hao-li/AIDev/pr_commit_details.parquet"
OUT_DIR = "data/interim/wf_rg"
COLS = ["pr_id", "sha", "filename", "status", "additions", "deletions"]
# Workflow files plus composite/local action definitions (they are part of CI configuration too).
PATTERN = r"(^|/)\.github/workflows/[^/]+\.ya?ml$|(^|/)\.github/actions/.+/action\.ya?ml$"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-groups", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    fs = HfFileSystem()
    with fs.open(SRC, "rb", block_size=8 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        n = pf.metadata.num_row_groups if args.max_groups is None else args.max_groups
        total_rows = total_hits = 0
        for g in range(n):
            out = f"{OUT_DIR}/{g:03d}.parquet"
            if os.path.exists(out):
                continue
            t = pf.read_row_group(g, columns=COLS)
            mask = pc.match_substring_regex(pc.fill_null(t["filename"], ""), PATTERN)
            hits = t.filter(mask)
            pq.write_table(hits, out)
            total_rows += t.num_rows
            total_hits += hits.num_rows
            print(f"rg {g + 1}/{n}: rows={t.num_rows:,} hits={hits.num_rows:,} "
                  f"cum_hits={total_hits:,}", flush=True)

    tables = [pq.read_table(f"{OUT_DIR}/{f}") for f in sorted(os.listdir(OUT_DIR))]
    allhits = pa.concat_tables(tables)
    os.makedirs("data/processed", exist_ok=True)
    pq.write_table(allhits, "data/processed/workflow_file_changes.parquet")
    print(f"DONE row_groups_saved={len(tables)} file_changes={allhits.num_rows:,} "
          f"distinct_prs={len(pc.unique(allhits['pr_id'])):,}")


if __name__ == "__main__":
    main()
