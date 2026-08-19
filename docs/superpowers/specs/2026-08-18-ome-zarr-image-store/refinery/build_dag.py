"""Derive the task DAG from each task's Files:/Interfaces: blocks.

Edges come from two sources:
  * file overlap -- task B edits a file task A creates
  * declared Consumes/Produces symbols
Shared files are recorded because they are what forbids parallelism.
"""
import re
from collections import defaultdict
from pathlib import Path

PLANS = Path(
    "/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store"
    "/docs/superpowers/plans/2026-08-18-ome-zarr-image-store"
)

TASK_RE = re.compile(r"^### (Task ([\d]+)\.([\w]+)):\s*(.*)$", re.M)
PATH_RE = re.compile(r"`([\w./_-]+\.(?:py|md|toml|yml|json))`")

tasks = {}  # id -> dict
for doc in sorted(PLANS.glob("phase-*.md")):
    body = doc.read_text()
    marks = list(TASK_RE.finditer(body))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[m.start():end]
        tid = m.group(1).replace("Task ", "")
        title = m.group(4).strip()
        cut = "CUT" in title or "~~" in title

        # Files: block -- everything up to the next bold heading
        files = {"create": set(), "modify": set(), "test": set(), "delete": set()}
        fm = re.search(r"\*\*Files:\*\*(.*?)(?:\n\*\*|\n- \[ \])", seg, re.S)
        if fm:
            for line in fm.group(1).splitlines():
                low = line.lower()
                for path in PATH_RE.findall(line):
                    if "create" in low:
                        files["create"].add(path)
                    elif "delete" in low:
                        files["delete"].add(path)
                    elif low.strip().startswith("- test") or "(extend)" in low:
                        files["test"].add(path)
                    elif "modify" in low or "extend" in low:
                        files["modify"].add(path)
                    else:
                        files["modify"].add(path)

        im = re.search(r"\*\*Interfaces:\*\*(.*?)(?:\n\*\*|\n- \[ \])", seg, re.S)
        interfaces = im.group(1).strip() if im else ""

        tasks[tid] = {
            "phase": m.group(2), "title": title, "cut": cut,
            "files": files, "interfaces": interfaces,
            "lines": len(seg.splitlines()), "doc": doc.name,
        }

live = {k: v for k, v in tasks.items() if not v["cut"]}
print(f"{len(tasks)} tasks ({len(live)} live, {len(tasks)-len(live)} cut)\n")

# ---- file -> tasks that touch it -----------------------------------------
touch = defaultdict(list)
for tid, t in live.items():
    for kind in ("create", "modify", "test", "delete"):
        for f in t["files"][kind]:
            touch[f].append((tid, kind))

print("=== FILES TOUCHED BY >1 TASK (these forbid parallelism) ===")
for f, ts in sorted(touch.items(), key=lambda kv: -len(kv[1])):
    if len(ts) > 1:
        print(f"  {f}")
        print(f"      {', '.join(f'{a}({b})' for a, b in sorted(ts))}")

print("\n=== PER-TASK SIZE AND FILE COUNT ===")
for tid in sorted(live, key=lambda x: (int(x.split('.')[0]), x)):
    t = live[tid]
    nf = sum(len(t["files"][k]) for k in t["files"])
    print(f"  {tid:<6} {t['lines']:>5} lines  {nf} files   {t['title'][:56]}")
