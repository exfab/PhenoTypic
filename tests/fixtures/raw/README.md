# Camera-RAW test fixtures

`synthetic_plate.dng` is a 64x64, 12-bit RGGB Bayer mosaic of a dark agar plate
with two bright colonies, stored as an uncompressed DNG 1.4 (6,624 bytes). It
was generated, not photographed: no camera image with known redistribution terms
was available, and a generated file has no license question at all.

- **Generator:** `docs/superpowers/plans/2026-09-24-cli-preflight/make_raw_fixture.py`
- **Command:** `uvx --with numpy --from pidng==4.0.9 python make_raw_fixture.py <out.dng>`
- **Deterministic:** the generator uses no randomness, so the bytes can be
  regenerated and compared.
- **Why it exists:** plan Task 9 (review R9). Before the RAW decoding fix the
  rawpy branch of `Image.imread` was unreachable, so its parameters had never
  run; this fixture is decoded by LibRaw exactly as a camera's DNG is.
