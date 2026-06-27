---
name: Review Changes
description: Perform a structured, impact-aware code review
---

## Review Changes

Perform a thorough, risk-aware code review using `git` + the CodeGraphContext graph.

### Steps

1. Run `git diff` (or `mcp__git__git_diff`) to get the changed files and functions.
2. For each changed symbol, use `analyze_code_relationships` to find
   callers/dependents (the blast radius).
3. Use `execute_cypher_query` to trace affected paths back to entry points.
4. Use `calculate_cyclomatic_complexity` / `find_most_complex_functions` to flag
   risky complexity introduced by the change.
5. For any untested changes, suggest specific test cases.

### Output Format

Group findings by risk level (high / medium / low) with:
- What changed and why it matters
- Impact radius (dependents)
- Suggested improvements
- Overall merge recommendation
