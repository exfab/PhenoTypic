# Vendored NGFF 0.5 JSON schemas

Read-only upstream reference material. **Never lint, format, autofix, or edit
these files.** They are the artifact every conformance assertion resolves
against; editing one silently invalidates every claim ever checked against it.

- Upstream: <https://ngff.openmicroscopy.org/0.5/schemas/>
- Retrieved: 2026-08-19

| file | sha256 |
|---|---|
| `image.schema` | `eb129f72fdd91e0da5c762b0214ac96a714a562c559fc60c6f688e28644c929d` |
| `label.schema` | `9f8bfd0cc0d50c6680dd39abe9b569e53283651933fac56d3b82ca8345740aec` |
| `ome.schema` | `9b9b3259256e928a2d2994f58fee47377eddbe8c0e421e50dfd2cca391e711c5` |
| `_version.schema` | `26c03746999987ae695a4d0efb2ae56503dd69b7b80320732f2a04a540ef91d5` |

Four facts about these files that the spec gets wrong or omits:

- `ome.schema` **requires** `["series", "version"]`, though the prose presents
  named series as optional.
- `label.schema` **requires** `["image-label", "version"]`, though the prose says
  SHOULD — **but `$defs/image-label` has no `required` list**, so `colors` is
  optional and nothing requires one entry per unique label value. The spec's
  §2.3 "MUST" is a PhenoTypic policy, not an NGFF rule.
- `$defs/omero` requires only `["channels"]`; the channel item has no `required`
  list and `color` is an unconstrained string. Only `window`, **if present**,
  requires all four of `start`/`min`/`end`/`max`. Emitting the full block is
  PhenoTypic policy too.
- All three reference `_version.schema` remotely, which is why it is vendored
  here and resolved through a `referencing.Registry` rather than fetched.
