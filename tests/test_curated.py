"""Tests for the curated collection split (mempalace/curated.py)."""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.curated import is_history, mirror_delete, mirror_upsert, sync
from mempalace.searcher import (
    ARCHIVE_COLLECTION,
    CURATED_COLLECTION,
    resolve_drawers_collection,
)


# ── is_history ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "meta",
    [
        {"wing": "claude_history", "source_file": "/notes/a.md"},
        {"wing": "claude-history", "source_file": "/notes/a.md"},
        {"wing": "personal", "source_file": "/AI/exports/convos_001.json"},
        {"wing": "personal", "source_file": r"C:\AI\stage\claude-history\s.md"},
        {"wing": "personal", "source_file": "/Downloads/claude_export/split/x.md"},
        {"wing": "personal", "source_file": "/stage/claude-history/sesiones-cowork/s.md"},
    ],
)
def test_is_history_true(meta):
    assert is_history(meta) is True


@pytest.mark.parametrize(
    "meta",
    [
        {"wing": "personal", "source_file": "/wiki/automation/note.md"},
        {"wing": "angola-cables", "source_file": "/dash/dashboard_data.js"},
        {"wing": "personal", "source_file": ""},
        {},
        None,
    ],
)
def test_is_history_false(meta):
    assert is_history(meta) is False


def test_is_history_matches_backslash_paths():
    """Windows paths must be normalised before marker matching."""
    assert is_history({"wing": "personal", "source_file": r"D:\ai\convos_002.json"}) is True


# ── resolve_drawers_collection ─────────────────────────────────────────


def test_resolve_prefers_curated_when_populated():
    curated = MagicMock()
    curated.count.return_value = 27638
    with patch("mempalace.searcher.get_collection", return_value=curated) as gc:
        col, name = resolve_drawers_collection("/fake")
    assert col is curated
    assert name == CURATED_COLLECTION
    assert gc.call_args.kwargs["collection_name"] == CURATED_COLLECTION


def test_resolve_falls_back_when_curated_missing():
    """A palace with no curated collection must keep working, unchanged."""
    archive = MagicMock()

    def fake_get(palace_path, collection_name=None, create=False):
        if collection_name == CURATED_COLLECTION:
            raise ValueError("does not exist")
        return archive

    with patch("mempalace.searcher.get_collection", side_effect=fake_get):
        col, name = resolve_drawers_collection("/fake")
    assert col is archive
    assert name == ARCHIVE_COLLECTION


def test_resolve_falls_back_when_curated_empty():
    """An empty curated collection is worse than none — do not search it."""
    empty = MagicMock()
    empty.count.return_value = 0
    with patch("mempalace.searcher.get_collection", return_value=empty):
        _, name = resolve_drawers_collection("/fake")
    assert name == ARCHIVE_COLLECTION


def test_resolve_scope_all_skips_curated():
    archive = MagicMock()
    with patch("mempalace.searcher.get_collection", return_value=archive) as gc:
        col, name = resolve_drawers_collection("/fake", scope="all")
    assert col is archive
    assert name == ARCHIVE_COLLECTION
    assert gc.call_count == 1  # curated never consulted


# ── sync ───────────────────────────────────────────────────────────────


def _paged(ids_by_page, payload=None):
    """Build a get() side effect that pages through ids_by_page."""
    calls = {"n": 0}

    def _get(**kwargs):
        if "ids" in kwargs:
            wanted = list(kwargs["ids"])
            return {
                "ids": wanted,
                "documents": [payload["documents"][i] for i in wanted],
                "metadatas": [payload["metadatas"][i] for i in wanted],
                "embeddings": [payload["embeddings"][i] for i in wanted],
            }
        page = ids_by_page[calls["n"]] if calls["n"] < len(ids_by_page) else []
        calls["n"] += 1
        return {"ids": page}

    return _get


def test_sync_reports_error_without_curated_collection():
    def fake_get(palace_path, collection_name=None, create=False):
        if collection_name == CURATED_COLLECTION:
            raise ValueError("does not exist")
        return MagicMock()

    with patch("mempalace.curated.get_collection", side_effect=fake_get):
        assert "error" in sync("/fake")


def test_sync_adds_new_and_drops_stale_skipping_history():
    payload = {
        "documents": {"new_ok": "note", "new_hist": "chat"},
        "metadatas": {
            "new_ok": {"wing": "personal", "source_file": "/wiki/n.md"},
            "new_hist": {"wing": "claude_history", "source_file": "/s.md"},
        },
        "embeddings": {"new_ok": [0.1] * 4, "new_hist": [0.2] * 4},
    }
    archive = MagicMock()
    archive.count.return_value = 3
    archive.get.side_effect = _paged([["keep", "new_ok", "new_hist"], []], payload)

    curated = MagicMock()
    curated.count.return_value = 2
    curated.get.side_effect = _paged([["keep", "gone"], []])

    def fake_get(palace_path, collection_name=None, create=False):
        return curated if collection_name == CURATED_COLLECTION else archive

    with patch("mempalace.curated.get_collection", side_effect=fake_get):
        stats = sync("/fake")

    assert stats == {"added": 1, "removed": 1, "skipped": 1}
    curated.delete.assert_called_once_with(ids=["gone"])
    added = curated.add.call_args.kwargs
    assert added["ids"] == ["new_ok"]
    # Vectors are copied, never recomputed.
    assert added["embeddings"] == [[0.1] * 4]


# ── mirrors ────────────────────────────────────────────────────────────


def test_mirror_upsert_copies_vector_and_skips_history():
    archive = MagicMock()
    archive.get.return_value = {
        "ids": ["a", "b"],
        "documents": ["note", "chat"],
        "metadatas": [
            {"wing": "personal", "source_file": "/wiki/n.md"},
            {"wing": "claude_history", "source_file": "/s.md"},
        ],
        "embeddings": [[0.5] * 4, [0.6] * 4],
    }
    curated = MagicMock()

    def fake_get(palace_path, collection_name=None, create=False):
        return curated if collection_name == CURATED_COLLECTION else archive

    with patch("mempalace.curated.get_collection", side_effect=fake_get):
        mirror_upsert("/fake", ["a", "b"])

    kw = curated.upsert.call_args.kwargs
    assert kw["ids"] == ["a"]
    assert kw["embeddings"] == [[0.5] * 4]


def test_mirrors_never_raise():
    """A broken mirror must not break the write that triggered it."""
    with patch("mempalace.curated.get_collection", side_effect=RuntimeError("boom")):
        mirror_upsert("/fake", ["a"])
        mirror_delete("/fake", ["a"])


def test_mirrors_noop_on_empty_ids():
    with patch("mempalace.curated.get_collection") as gc:
        mirror_upsert("/fake", [])
        mirror_delete("/fake", [])
    gc.assert_not_called()
