# Vendored OME-XML 2016-06 schema

Read-only upstream reference material. **Never lint, format, autofix, or edit
this file.** It is the artifact the OME-XML conformance assertion resolves
against; editing it silently invalidates every claim ever checked against it.

- Upstream: <http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd>
- Retrieved: 2026-08-19
- Target namespace: `http://www.openmicroscopy.org/Schemas/OME/2016-06`

| file | sha256 |
|---|---|
| `ome.xsd` | `64b439ff488c87d81ca112b73b7123596952ff8a8543e3b02d94ea8db5ed51ee` |

Why this file is vendored, and why one file is enough:

- NGFF 0.5 §2.2.3 makes `OME/METADATA.ome.xml` a conditional MUST — it *"MUST
  adhere to the OME-XML specification but MUST use `<MetadataOnly/>`
  elements"* — so the store's OME-XML is validated, not merely emitted
  (ledger **ALGO-1**).
- `xmlschema` is the validator. The dependency is declared in the `dev`
  dependency group by Task 0.1, not left transitive: spec §7 forbids a
  conformance check that skips on a missing dependency.
- `ome.xsd` carries exactly **one** remote `xsd:import` (line 30,
  `http://www.w3.org/2001/xml.xsd`), which resolves against `xmlschema`'s
  bundled fallback locations rather than the network. Vendoring this single
  file is therefore sufficient offline.
