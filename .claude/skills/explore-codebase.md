---
name: Explore Codebase
description: Navigate and understand codebase structure using the knowledge graph
---

## Explore Codebase

Use the CodeGraphContext MCP tools to explore and understand the codebase.

### Steps

1. Run `get_repository_stats` for overall codebase metrics and structure.
2. Use `list_indexed_repositories` to confirm what is indexed.
3. Use `find_code` to locate specific functions, classes, or variables.
4. Use `analyze_code_relationships` to trace callers, callees, imports, and
   dependencies of a symbol.
5. Use `execute_cypher_query` for arbitrary multi-hop relationship traversal.
6. Use `find_most_complex_functions` to spot complexity hotspots.

### Tips

- Start broad (`get_repository_stats`) then narrow with `find_code`.
- Map a symbol's neighborhood with `analyze_code_relationships` before opening files.
- If results look stale, re-index with `add_code_to_graph` (auto-watch is off).
