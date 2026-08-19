"""Post-round-3 sweep. Extends selfcheck2 with the checks that would have caught
three of round 3's findings mechanically (simp-r3's recommendation):

  * fixture identifiers named in exit criteria vs. fixtures any task defines (SIMP-20)
  * blocks landing under the WRONG TASK -- the dominant defect class of this pass
  * stale pass-1/pass-2 numbering after the MIG-15 inversion
"""
import re
from pathlib import Path

BASE = Path(
    "/bigdata/exfab/anguy344/PhenoTypic/.worktrees/worktree-ome-zarr-image-store"
    "/docs/superpowers"
)
PLANS = BASE / "plans/2026-08-18-ome-zarr-image-store"
SPEC = BASE / "specs/2026-08-18-ome-zarr-image-store/design.md"

DOCS = {p.name: p.read_text() for p in sorted(PLANS.glob("*.md"))}
DOCS[SPEC.name] = SPEC.read_text()
HISTORY = {"OPEN-QUESTIONS.md"}

fails, notes = [], []


def check(label, ok, detail=""):
    (notes if ok else fails).append(
        f"{'PASS' if ok else 'FAIL'}  {label}" + (f" -- {detail}" if detail else "")
    )


def hits(pattern, *, skip_quotes=True):
    out, rx = [], re.compile(pattern)
    for name, body in DOCS.items():
        if name in HISTORY:
            continue
        for i, line in enumerate(body.splitlines(), 1):
            s = line.lstrip()
            if skip_quotes and (s.startswith(">") or s.startswith("#")):
                continue
            if rx.search(line):
                out.append(f"{name}:{i}")
    return out


# ---- removed / renamed symbols -------------------------------------------
for sym in ("DOWNSAMPLE_DESCRIPTIONS", "_check_tunable_levels", "BundleLayout.resolve",
            "migrate_metadata_schema\\b", "v2_rich", "test_jsonschema_is_declared"):
    h = hits(rf"{sym}")
    check(f"gone: {sym}", not h, ", ".join(h[:4]))

# ---- SIMP-20: fixture identifiers in exit criteria must be defined --------
FIXTURES = set()
for body in DOCS.values():
    FIXTURES |= set(re.findall(r"^\s*\|\s*`(\w+)`\s*\|", body, re.M))
    FIXTURES |= set(re.findall(r"^def (\w+)\(", body, re.M))
    FIXTURES |= set(re.findall(r'parametrize\(\s*"\w+",\s*\[([^\]]*)\]', body))
GOLDENS = set(re.findall(r"\b(v1_\w+|v2_\w+)\b", " ".join(DOCS.values())))
for name, body in DOCS.items():
    if "exit criteria" not in body:
        continue
    tail = body.split("exit criteria", 1)[1]
    for gold in set(re.findall(r"`(v1_\w+|v2_\w+)`", tail)):
        defined = sum(body.count(gold) for body in DOCS.values())
        check(f"{name} exit criteria fixture `{gold}` is defined elsewhere",
              defined > 1, f"appears {defined}x in the whole plan")

# ---- pass numbering after the MIG-15 inversion ---------------------------
h = hits(r"pass 2 rewrites|pass 2 needs them|pass 2 does the same thing|"
         r"Without pass 2 those targets")
check("no stale pre-inversion pass numbering", not h, ", ".join(h[:4]))

# ---- wrong-task placement: every Step 3a/Files block under its own task ---
def task_of(body, idx):
    lines = body.splitlines()
    return next((x for x in reversed(lines[:idx]) if x.startswith("### Task ")), None)


for name, body in DOCS.items():
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if "tests/_ngff_conformance.py`**" in line and "Step 3a" in line:
            owner = task_of(body, i) or ""
            check(f"{name}: Step 3a under Task 1.4", owner.startswith("### Task 1.4"),
                  f"under {owner[:34]}")

# ---- each phase's Files: blocks precede its steps -------------------------
for name, body in DOCS.items():
    if not name.startswith("phase-"):
        continue
    for m in re.finditer(r"^### (Task [\d.a-z]+):", body, re.M):
        start = m.end()
        nxt = body.find("\n### Task ", start)
        seg = body[start: nxt if nxt != -1 else len(body)]
        if "**Files:**" not in seg:
            continue
        if "- [ ] **Step" in seg and seg.index("**Files:**") > seg.index("- [ ] **Step"):
            fails.append(f"FAIL  {name} {m.group(1)}: Files: block comes after its first Step")

# ---- README task counts --------------------------------------------------
readme = DOCS["README.md"]
for name, body in DOCS.items():
    if not name.startswith("phase-"):
        continue
    num = name.split("-")[1]
    actual = len(re.findall(rf"^### Task {num}\.", body, re.M))
    row = re.search(rf"\[`{re.escape(name)}`\]\([^)]*\)\s*\|\s*(\d+)", readme)
    if row:
        check(f"README count {name}", int(row.group(1)) == actual,
              f"README {row.group(1)} vs {actual}")

print("\n".join(notes))
print()
print("\n".join(fails) if fails else "ALL CHECKS PASS")
print(f"\n{len(fails)} failure(s)")
