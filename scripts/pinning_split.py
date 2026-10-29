"""Split the supply-chain outcome by action owner (first-party major tags are GitHub's own recommendation).

For every `uses:` reference to a repository action (owner/repo[/path]@ref; local ./ and docker:// excluded):
  first-party = owner in {actions, github};  pinned = ref is a full 40-hex commit SHA.
Per file: unpinned_1p (any first-party ref not SHA-pinned), unpinned_3p (any third-party ref not SHA-pinned),
uses_3p (any third-party action at all). Files: agent D1 drafts + D2 finals and human added files.
Output: results/pinning_flags.csv
"""
import os
import re

import pandas as pd

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*['\"]?([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)(?:/[^@\s'\"]*)?@([^\s'\"#]+)",
                  re.M)
SHA = re.compile(r"^[0-9a-f]{40}$")
FIRST = {"actions", "github"}


def flags(text):
    u1 = u3 = any3 = False
    for owner, _repo, ref in USES.findall(text):
        third = owner.lower() not in FIRST
        any3 |= third
        if not SHA.match(ref):
            if third:
                u3 = True
            else:
                u1 = True
    return {"unpinned_1p": u1, "unpinned_3p": u3, "uses_3p": any3}


def main():
    v = pd.read_parquet("data/processed/versions.parquet")
    a = v[(v.origin == "added") & ((v.ver == 0) | ((v.ver == v.n_ver - 1) & v.merged))]
    a = a.drop_duplicates(["sha", "filename"])[["sha", "filename"]].assign(root="data/files")
    h = pd.read_parquet("data/processed/human_wf_changes.parquet")
    h = h[h.status == "added"].drop_duplicates(["sha", "filename"])[["sha", "filename"]].assign(root="data/files_human")
    rows = []
    for r in pd.concat([a, h]).itertuples():
        p = os.path.join(r.root, r.sha, r.filename)
        text = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
        rows.append({"sha": r.sha, "filename": r.filename, **flags(text)})
    d = pd.DataFrame(rows)
    d.to_csv("results/pinning_flags.csv", index=False)
    print(d[["unpinned_1p", "unpinned_3p", "uses_3p"]].mean().round(3))


if __name__ == "__main__":
    main()
