"""``directory_digest`` — a stable identity for a whole image directory.

Nothing shipped before this could identify an image *set*:
``bytes_fingerprint``/``file_fingerprint`` are single-file and
``paths_fingerprint`` needs the caller to have enumerated the paths already.
Without a directory-level digest, ``campaign_status.comparable`` cannot detect
two arms tuned against different image sets, and a subset artifact has no
``parent.digest`` to verify a promotion against.

The three properties that make it usable as an identity are asserted here:
the ``sha256:`` prefix (so it string-compares against ``file_fingerprint``),
sensitivity to the file set, and independence from filesystem listing order.
"""

from __future__ import annotations

import pytest


def test_digest_is_stable_and_prefixed(tmp_path):
    from phenotypic.sdk_._io_constants import directory_digest

    (tmp_path / "a.tif").write_bytes(b"aaa")
    (tmp_path / "b.tif").write_bytes(b"bbb")

    first = directory_digest(tmp_path)
    assert first.startswith("sha256:")
    assert first == directory_digest(tmp_path)


def test_digest_changes_when_a_file_is_added(tmp_path):
    from phenotypic.sdk_._io_constants import directory_digest

    (tmp_path / "a.tif").write_bytes(b"aaa")
    before = directory_digest(tmp_path)
    (tmp_path / "c.tif").write_bytes(b"ccc")
    assert directory_digest(tmp_path) != before


def test_digest_changes_when_a_file_grows(tmp_path):
    """Size is in the digest **on its own**.

    Rewriting a file also moves its ``mtime_ns``, so the timestamp is pinned
    back to its old value first. Without that, this test passes against an
    implementation that hashes no size at all.
    """
    import os

    from phenotypic.sdk_._io_constants import directory_digest

    target = tmp_path / "a.tif"
    target.write_bytes(b"aaa")
    original = target.stat()
    before = directory_digest(tmp_path)

    target.write_bytes(b"aaaa")
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert target.stat().st_mtime_ns == original.st_mtime_ns

    assert directory_digest(tmp_path) != before


def test_digest_changes_when_only_the_mtime_moves(tmp_path):
    """``mtime_ns`` is in the digest **on its own**.

    It is the only term that can catch an in-place rewrite preserving both the
    path and the byte count — a plate re-exported at the same dimensions and
    compression. The cost of carrying it is stated in the docstring: a copy
    made without ``-p`` reads as a different image set.
    """
    import os

    from phenotypic.sdk_._io_constants import directory_digest

    target = tmp_path / "a.tif"
    target.write_bytes(b"aaa")
    before = directory_digest(tmp_path)

    moved = target.stat().st_mtime_ns + 1_000_000_000
    os.utime(target, ns=(moved, moved))
    assert target.stat().st_size == 3

    assert directory_digest(tmp_path) != before


def test_digest_ignores_listing_order(tmp_path, monkeypatch):
    """Sorted by relative path, so filesystem order cannot leak in.

    The walk itself is reversed rather than the entry list, so this fails
    against an implementation that drops the sort -- which is the only thing
    standing between the digest and the order ``scandir`` happens to return.
    """
    from phenotypic.sdk_ import _io_constants

    (tmp_path / "z.tif").write_bytes(b"z")
    (tmp_path / "a.tif").write_bytes(b"a")
    forward = _io_constants.directory_digest(tmp_path)

    real_walk = _io_constants.os.walk

    def _reversed_walk(top, **kwargs):
        for directory, subdirectories, filenames in reversed(
            list(real_walk(top, **kwargs))
        ):
            yield directory, subdirectories, list(reversed(filenames))

    monkeypatch.setattr(_io_constants.os, "walk", _reversed_walk)
    assert _io_constants.directory_digest(tmp_path) == forward


def test_an_unlistable_subdirectory_is_refused_not_skipped(tmp_path):
    """An identity that varies by who reads it is worse than an error.

    ``rglob`` swallows the ``PermissionError``, so an unreadable subtree used
    to contribute nothing and the digest simply came out different -- two
    users getting two identities for one tree, with nothing to say why. Since
    the digest decides "is this resume compatible" and "does the human's
    approval still apply", that is a silent wrong answer.
    """
    import os

    from phenotypic.sdk_._io_constants import directory_digest

    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")

    (tmp_path / "readable").mkdir()
    (tmp_path / "readable" / "a.tif").write_bytes(b"aaa")
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "b.tif").write_bytes(b"bbb")
    closed.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            directory_digest(tmp_path)
    finally:
        closed.chmod(0o700)


def test_relative_to_a_non_ancestor_is_refused(tmp_path):
    """The silent fallback recorded **absolute** names.

    That is exactly what
    ``test_digest_is_relative_so_a_moved_directory_keeps_its_identity``
    exists to prevent, reached through a different door: the digest would
    become mount-dependent with nothing raised.
    """
    from phenotypic.sdk_._io_constants import directory_digest

    parent = tmp_path / "plates"
    parent.mkdir()
    (parent / "a.tif").write_bytes(b"aaa")
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()

    with pytest.raises(ValueError, match="ancestor"):
        directory_digest(parent, relative_to=unrelated)


def test_a_pre_1970_mtime_is_digestible(tmp_path):
    """A mis-set camera clock or a restored archive is not a crash.

    ``to_bytes(16, "big")`` refuses a negative int, so a file dated before the
    epoch raised ``OverflowError`` from inside an identity function.
    """
    import os

    from phenotypic.sdk_._io_constants import directory_digest

    target = tmp_path / "a.tif"
    target.write_bytes(b"aaa")
    os.utime(target, ns=(-1_000_000_000, -1_000_000_000))
    if target.stat().st_mtime_ns >= 0:
        pytest.skip("this filesystem cannot store a pre-1970 mtime")

    assert directory_digest(tmp_path).startswith("sha256:")


def test_each_name_is_length_prefixed_in_the_digest_stream(tmp_path):
    """Pins the framing that makes the entry stream unambiguous.

    Without the prefix, ``name || size || mtime`` records concatenate with no
    boundary, so a longer name and a shorter one followed by leading size
    bytes can serialize identically. No *realizable* filesystem state
    exhibits the collision -- it needs a size or timestamp above 2**56 -- so
    the property cannot be tested by constructing one, and the encoding is
    pinned directly instead. Recomputed here from the documented terms, not
    imported from the implementation.
    """
    import hashlib

    from phenotypic.sdk_._io_constants import directory_digest

    target = tmp_path / "plateA.tif"
    target.write_bytes(b"aaa")
    stat = target.stat()

    expected = hashlib.sha256()
    name = b"plateA.tif"
    expected.update(len(name).to_bytes(8, "big"))
    expected.update(name)
    expected.update(stat.st_size.to_bytes(8, "big"))
    expected.update(stat.st_mtime_ns.to_bytes(16, "big", signed=True))

    assert directory_digest(tmp_path) == f"sha256:{expected.hexdigest()}"


def test_digest_is_relative_so_a_moved_directory_keeps_its_identity(tmp_path):
    """Absolute paths must not enter the digest.

    A parent copied to a second cluster path is the same image set; a digest
    keyed on absolute paths would report every promotion as a changed dataset.
    """
    from phenotypic.sdk_._io_constants import directory_digest

    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "plateA").mkdir(parents=True)
        (root / "plateA" / "a.tif").write_bytes(b"aaa")

    # ``mtime_ns`` is in the digest, so equalize it the way a preserving copy
    # (``cp -p`` / ``rsync -a``) would.
    stat = (left / "plateA" / "a.tif").stat()
    import os

    os.utime(right / "plateA" / "a.tif", ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert directory_digest(left) == directory_digest(right)


def test_digest_sees_a_file_moved_between_datasets(tmp_path):
    """Parent-relative paths are in the digest, not just the file contents.

    ``plateA/x.tif`` and ``plateB/x.tif`` are different rows in the output
    (``Metadata_Dataset`` comes from the subdirectory name), so a restructured
    parent is a changed parent even when every byte is identical.
    """
    from phenotypic.sdk_._io_constants import directory_digest

    (tmp_path / "plateA").mkdir()
    (tmp_path / "plateB").mkdir()
    moved = tmp_path / "plateA" / "x.tif"
    moved.write_bytes(b"xxx")
    before = directory_digest(tmp_path)

    destination = tmp_path / "plateB" / "x.tif"
    moved.replace(destination)
    import os

    stat_before = destination.stat()
    os.utime(destination, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))

    assert directory_digest(tmp_path) != before


def test_relative_to_anchors_the_recorded_names(tmp_path):
    """``relative_to`` changes the recorded names, hence the digest."""
    from phenotypic.sdk_._io_constants import directory_digest

    parent = tmp_path / "plates"
    (parent / "plateA").mkdir(parents=True)
    (parent / "plateA" / "a.tif").write_bytes(b"aaa")

    assert directory_digest(parent) != directory_digest(parent, relative_to=tmp_path)


def test_missing_directory_raises(tmp_path):
    from phenotypic.sdk_._io_constants import directory_digest

    with pytest.raises(FileNotFoundError):
        directory_digest(tmp_path / "nope")
