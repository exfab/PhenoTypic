"""Summarize `python -X importtime` stderr: total, then heaviest modules by cumulative time."""
import re, sys
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|(\s*)(\S+)", line)
    if m:
        rows.append((int(m.group(1)), int(m.group(2)), (len(m.group(3)) - 1) // 2, m.group(4)))
root = sys.argv[2]
total = max((c for s, c, d, n in rows if n == root), default=0)
print(f"{root}: cumulative {total/1e6:.2f} s over {len(rows)} modules")
by_top = {}
for s, c, d, n in rows:
    top = n.split(".")[0]
    by_top[top] = by_top.get(top, 0) + s
print("  self time by top-level package (top 18):")
for top, s in sorted(by_top.items(), key=lambda kv: -kv[1])[:18]:
    print(f"    {s/1e6:6.2f} s  {top}")
print("  heaviest phenotypic modules by cumulative (top 15, depth<=6):")
for s, c, d, n in sorted((r for r in rows if r[3].startswith("phenotypic")), key=lambda r: -r[1])[:15]:
    print(f"    {c/1e6:6.2f} s  {'  '*min(d,6)}{n}")
