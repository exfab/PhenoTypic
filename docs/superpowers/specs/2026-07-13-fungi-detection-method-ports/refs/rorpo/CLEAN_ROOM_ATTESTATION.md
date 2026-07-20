# A08 clean-room attestation

## Export preparation

- Oracle agent: `/root/implement_a02_tensor`
- Export mechanism: `prepare_clean_room.py` from the reviewed evidence commit
- Restricted source, archive, adapter, executable, and raw capture included: **no**
- `.git` directory or Git object database included: **no**
- A08-specific inputs limited to `CLEAN_ROOM_ALLOWLIST.txt`: **yes**
- Root control manifest generated with the full excluded Git corpus: **yes**

## Production implementer attestation

Complete before any production code is authored:

- Implementer identity: `/root/implement_a01_gwdt`
- Sanitized export source commit: `115339d3d85968db43e508ecc1f531c5d2199c2a`
- Sanitized export manifest SHA-256:
  `ff3d918a40c3caeb92aa310498984faf444fff533766ed7658fe4f66321bbefe`
- I received only the generated sanitized export: **yes**
- I did not inspect any path listed in the manifest's `forbidden_git_corpus`: **yes**
- I did not inspect `/private/tmp/RORPO2D.zip`,
  `/private/tmp/RORPO2D-ipol-1.0-restricted`, `/private/tmp/rorpo_oracle_adapter.cpp`,
  `/private/tmp/rorpo_oracle_adapter`, or `/private/tmp/rorpo_fixture_raw_v2.json`: **yes**

## Independent reviewer verification

- Reviewer identity: `/root/implement_a02_tensor`
- Evidence commit reviewed: `dd0c966aadcf7b2a152b392b0fcf2a228e575177`
- Export manifest verified:
  `ff3d918a40c3caeb92aa310498984faf444fff533766ed7658fe4f66321bbefe`
- Reachable Git objects scanned for restricted members: **yes; zero restricted hashes found**
- G0 status: **PASS** at `dd0c966aadcf7b2a152b392b0fcf2a228e575177`
- G3 production commit reviewed: `618222878671fde30291c552bb1c48c67a8f2a16`
- G3 status: **PASS** with zero findings
