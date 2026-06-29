---
name: Debug Issue
description: Systematically debug issues using graph-powered code navigation
---

## Debug Issue

Use the CodeGraphContext knowledge graph to systematically trace and debug issues.

### Steps

1. Use `find_code` to locate the code related to the issue (by name or keyword).
2. Use `analyze_code_relationships` to trace callers/callees and dependencies of
   the suspect symbol.
3. Use `execute_cypher_query` for deeper multi-hop traversal (e.g. the full call
   chain from an entry point to the failing function).
4. Use `git diff` / `git log` (or `mcp__git__git_diff`) to check whether a recent
   change introduced the issue.
5. Use `analyze_code_relationships` on the suspected files to see what else
   depends on them.

### Tips

- Check both callers and callees to understand the full context.
- Recent changes are the most common source of new issues.
- If the graph looks stale, re-index with `add_code_to_graph` (auto-watch is off).
