"""Tests for mempalace.console — stdio encoding for legacy Windows code pages."""

import io
import sys

from mempalace.console import _is_utf8, configure_stdio

# The glyphs the CLI prints everywhere: box-drawing, check, cross, arrow.
GLYPHS = "─│ ✓ ✗ → —"


def _wrapper(encoding):
    """A real TextIOWrapper (so it has .reconfigure) over an in-memory buffer."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")


# ── _is_utf8 ───────────────────────────────────────────────────────────


def test_is_utf8_recognizes_aliases():
    assert _is_utf8(_wrapper("utf-8"))
    assert _is_utf8(_wrapper("UTF8"))
    assert _is_utf8(_wrapper("utf_8"))


def test_is_utf8_false_for_legacy_and_unknown():
    assert not _is_utf8(_wrapper("cp1252"))
    assert not _is_utf8(io.StringIO())  # no .encoding attribute
    assert not _is_utf8(type("S", (), {"encoding": "not-a-codec"})())


# ── configure_stdio ────────────────────────────────────────────────────


def test_cp1252_stdout_can_print_glyphs(monkeypatch):
    """The regression: printing box-drawing on a cp1252 stdout used to raise."""
    stdout = _wrapper("cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    configure_stdio()
    print(GLYPHS)  # UnicodeEncodeError before the fix
    stdout.flush()

    assert stdout.encoding.lower().replace("-", "") == "utf8"
    assert stdout.buffer.getvalue().decode("utf-8").strip() == GLYPHS


def test_cp1252_stderr_and_stdin_are_converted(monkeypatch):
    stderr, stdin = _wrapper("cp1252"), _wrapper("cp1252")
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "stdin", stdin)

    configure_stdio()

    assert _is_utf8(stderr)
    assert _is_utf8(stdin)


def test_utf8_stream_is_left_alone(monkeypatch):
    stdout = _wrapper("utf-8")
    monkeypatch.setattr(sys, "stdout", stdout)

    configure_stdio()

    # Still strict — we never touched it, so the host's choices survive.
    assert stdout.errors == "strict"


def test_unencodable_glyph_degrades_instead_of_raising(monkeypatch):
    stdout = _wrapper("cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    configure_stdio()
    print("\ud800")  # lone surrogate: not encodable even in UTF-8
    stdout.flush()

    assert stdout.buffer.getvalue()  # replaced, not raised


def test_non_reconfigurable_streams_are_tolerated(monkeypatch):
    """Under pytest capture sys.stdout is not a TextIOWrapper."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", None)

    configure_stdio()  # must not raise


def test_is_idempotent(monkeypatch):
    stdout = _wrapper("cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    configure_stdio()
    configure_stdio()

    assert _is_utf8(stdout)
