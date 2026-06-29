---
name: Refactor Safely
description: Plan and execute safe refactoring using dependency analysis
---

## Refactor Safely

Use the CodeGraphContext knowledge graph to plan refactoring with confidence.

### Steps

1. Use `find_dead_code` to locate unreferenced functions/classes safe to remove.
2. Use `find_code` to locate the symbol you intend to change.
3. Use `analyze_code_relationships` (escalate to `execute_cypher_query` for deep
   reach) to enumerate every caller/dependent before renaming or changing a
   signature.
4. Apply edits with the normal Edit tools, using that dependent list as your
   checklist.
5. After changes, re-index with `add_code_to_graph` and re-run
   `analyze_code_relationships` to confirm nothing dangles.

### Safety Checks

- Enumerate all callers/dependents before editing — `analyze_code_relationships`
  is your impact map.
- Use `find_most_complex_functions` to identify decomposition targets.
- CodeGraphContext has no rename/apply tool — perform edits manually and verify
  against the graph afterward.
