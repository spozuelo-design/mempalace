"""Console encoding helpers.

MemPalace prints box-drawing characters, check marks and arrows all over the
CLI. On Windows the standard streams fall back to the legacy ANSI code page
(cp1252 on most Western installs) whenever they are not attached to a real
console — a pipe, a redirect, a scheduled task, an MCP client — and every one
of those glyphs raises ``UnicodeEncodeError``.

The streams are reconfigured from the entry points (``cli.main`` and
``mcp_server.main``), never at import time, so importing ``mempalace`` as a
library leaves the host application's streams exactly as it set them up.
"""

import codecs
import sys


def _is_utf8(stream):
    """True if the stream already encodes as UTF-8."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def configure_stdio():
    """Force UTF-8 on stdin/stdout/stderr when the platform default isn't.

    Uses ``errors="replace"`` so an unexpected glyph degrades to ``?`` instead
    of killing the process mid-run. Streams that cannot be reconfigured (a
    plain ``StringIO`` under pytest capture, an already-read stdin) are left
    untouched. Safe to call more than once.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is None or _is_utf8(stream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # UnsupportedOperation (already-buffered stdin) or a stream that
            # rejects the arguments — printing ASCII still works, carry on.
            pass
