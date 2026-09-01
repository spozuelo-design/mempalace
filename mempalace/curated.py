"""
curated.py — the curated drawer collection.

`mempalace_drawers` holds everything that was ever mined, raw chat transcripts
included. On a palace where transcripts dominate they win every query: they are
semantically close to any technical question, so curated notes never surface.

Excluding them with a ``where`` clause does not work. ChromaDB (1.5.x) resolves
a filtered vector query by finding the HNSW neighbours *first* and intersecting
with the metadata filter afterwards. When 80% of the index is transcripts, the
neighbours are transcripts, and the intersection is small or empty — a filtered
query returns far fewer hits than asked for, sometimes zero, with no error.

So the split is physical: ``mempalace_curated`` is a second collection holding
only the non-transcript drawers. Queries against it are unfiltered, so HNSW
recall is intact. Nothing is deleted — ``mempalace_drawers`` remains the full
archive and stays reachable via scope="all".

Vectors are COPIED, never recomputed: ``get(include=["embeddings"])`` on the
archive and ``add(embeddings=...)`` on the curated collection. Rebuilding 138k
drawers this way takes ~2 minutes, against ~3 hours for a `repair` that
re-embeds.
"""

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .palace import get_collection

logger = logging.getLogger("mempalace")

CURATED_COLLECTION = "mempalace_curated"
ARCHIVE_COLLECTION = "mempalace_drawers"

# Wings that hold raw session transcripts rather than curated knowledge.
HISTORY_WINGS = frozenset({"claude_history", "claude-history"})

# Path fragments that mark a transcript dump even when it was filed under
# another wing (e.g. convos_001.json living in wing "personal").
HISTORY_PATH_MARKERS = (
    "convos_",
    "claude_export",
    "sesiones-cowork",
    "/claude-history/",
    "/claude_history/",
)

# Payload batches stay small — each row carries a 384-float vector.
_BATCH = 1000
# Id-only scans carry no payload, so page much bigger: the sync diff over 138k
# drawers goes from ~30s to a couple of seconds.
_ID_BATCH = 20000


# --- Data dumps -----------------------------------------------------------
#
# Excluding transcripts was not enough. An audit on 2026-08-06 found the curated
# collection was 80% machine-generated data: one file,
# dashboard-comercial/data/dashboard_data.js, held 12638 of 27687 drawers (46%),
# while genuine notes (.md) were 10.9%. A data dump is semantically close to any
# question about its subject, so it crowds out notes exactly the way transcripts
# used to — the failure that motivated splitting the collection, reproduced one
# level down.
#
# The tell is fan-out, not path: a document yields a handful of drawers, a dump
# yields hundreds. Keyed on extension too, so a long hand-written .md is never
# mistaken for a dump.
DUMP_EXTENSIONS = frozenset(
    {".js", ".json", ".jsonl", ".csv", ".tsv", ".html", ".htm", ".xml", ".map"}
)
# Drawers from a single source file above which it is machine-generated data.
DUMP_FANOUT = 150


def is_history(meta: dict) -> bool:
    """True if this drawer is raw conversation transcript, not curated knowledge."""
    if not meta:
        return False
    if meta.get("wing") in HISTORY_WINGS:
        return True
    src = (meta.get("source_file") or "").replace("\\", "/").lower()
    return any(marker in src for marker in HISTORY_PATH_MARKERS)


def _has_dump_extension(source_file: str) -> bool:
    name = (source_file or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return "." in name and ("." + name.rsplit(".", 1)[-1]) in DUMP_EXTENSIONS


def dump_sources(archive, fanout: int = DUMP_FANOUT) -> set:
    """Source files that exploded into so many drawers they must be data dumps.

    One metadata-only pass over the archive. Returns a set of source_file values
    to exclude wholesale.
    """
    counts = {}
    for page in _iter_archive(archive, ["metadatas"], batch=5000):
        for meta in page.get("metadatas") or []:
            src = (meta or {}).get("source_file")
            if src and _has_dump_extension(src):
                counts[src] = counts.get(src, 0) + 1
    return {src for src, n in counts.items() if n >= fanout}


def is_excluded(meta: dict, dumps: set) -> bool:
    """True if this drawer stays out of the curated collection."""
    if is_history(meta):
        return True
    return bool(dumps) and (meta or {}).get("source_file") in dumps


def _read_records(archive, offset, n, include, unreadable):
    """Read *n* records one by one, skipping the ones that cannot be read.

    A page can fail for reasons that have nothing to do with the drawer being
    lost: the vector segment may not have caught up with the write log yet, or
    an id may carry a tombstone from an earlier delete. Aborting the whole
    rebuild for that leaves the curated collection truncated, which is worse
    than finishing without those few records.

    Always reports *n* positions consumed, so paging by offset stays aligned
    even when some records drop out.
    """
    page = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    for k in range(offset, offset + n):
        try:
            one = archive.get(limit=1, offset=k, include=include)
        except Exception:
            unreadable.append(k)
            continue
        for key in page:
            got = one.get(key)
            if got is not None and len(got):
                page[key].append(got[0])
    return page, n


def _iter_archive(archive, include, batch=_BATCH, unreadable=None):
    """Page through the archive collection yielding raw get() payloads.

    Tolerant by design: if a page cannot be read, it drops to one record at a
    time and skips only what is genuinely unreadable. The offsets of what was
    skipped land in *unreadable* so the caller can name them instead of
    swallowing the loss.
    """
    if unreadable is None:
        unreadable = []
    offset, total = 0, archive.count()
    while offset < total:
        n = min(batch, total - offset)
        try:
            page = archive.get(limit=n, offset=offset, include=include)
            consumed = len(page.get("ids") or [])
            if consumed == 0:
                return
        except Exception as exc:
            logger.warning(
                "curated: unreadable page at offset %d (%s) - falling back to one record at a time",
                offset,
                exc,
            )
            page, consumed = _read_records(archive, offset, n, include, unreadable)
        if page.get("ids"):
            yield page
        offset += consumed


@contextmanager
def _rebuild_lock(palace_path: str, stale_after_s: int = 1800):
    """One rebuild at a time.

    Two rebuilds at once corrupt the curated collection: measured 2026-08-31,
    one deleted and recreated it while the other was still writing and the
    result was 9.080 drawers instead of ~11.850. There was no lock. Now there is.
    """
    lock = Path(palace_path) / ".curate-rebuild.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            quien = lock.read_text(encoding="utf-8").strip()
            edad = time.time() - lock.stat().st_mtime
        except Exception:
            quien, edad = "(ilegible)", 0.0
        if edad < stale_after_s:
            raise RuntimeError(
                "another rebuild holds the lock ({}), {:.0f}s old. Wait for it: "
                "it rebuilds from the archive, so its result also picks up what "
                "you just mined. Do not kill it and do not start another. "
                "If that process is gone -- a rebuild that segfaults leaves the "
                "lock behind, because a crash does not run the cleanup -- delete "
                "{} and try again; after {:.0f}s it is taken as stale on its "
                "own.".format(quien, edad, lock, stale_after_s)
            )
        logger.warning("curated: taking a stale rebuild lock (%s, %.0fs old)", quien, edad)
        lock.unlink(missing_ok=True)
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(
            fd,
            "pid={} started={}".format(
                os.getpid(), datetime.now().isoformat(timespec="seconds")
            ).encode(),
        )
        os.close(fd)
        yield
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            logger.warning("curated: could not remove %s", lock)


def rebuild(palace_path: str, ef_search: int = 400, progress=None) -> dict:
    """Rebuild the curated collection from the archive, without destroying it first.

    Builds into a temporary collection and only swaps names once it is complete.
    An interrupted rebuild therefore leaves the previous curated collection
    intact: worse than the ideal one, but whole and known. The old behaviour --
    delete first, refill after -- left `search` degraded at whatever point it
    died, measured 2026-09-01 at 10.708 drawers instead of ~11.950.

    Returns {"kept": n, "skipped": n, "unreadable": n}.
    """
    from .backends.chroma import ChromaBackend

    backend = ChromaBackend()

    # El lock va PRIMERO, antes de tocar el archivo. Tenerlo despues de
    # dump_sources() era un fallo: el segundo proceso escaneaba los ~141k
    # metadatos en paralelo con el primero y la maquina (8 GB soldados) moria
    # con un segfault antes de llegar a mirar el lock. Medido el 2026-09-01: dos
    # rebuilds a la vez, el segundo con rc=139 en vez del aviso. Un guardarrail
    # que se toma tarde no guarda nada.
    with _rebuild_lock(palace_path):
        archive = get_collection(palace_path, collection_name=ARCHIVE_COLLECTION, create=False)

        # Cheap metadata pass first: which source files are data dumps.
        dumps = dump_sources(archive)
        logger.info("curated rebuild: excluding %d data-dump source files", len(dumps))

        client = backend._client(palace_path)  # noqa: SLF001 — need configuration= support
        tmp_name = CURATED_COLLECTION + "_rebuilding"
        # A temp left behind by a crashed run must not poison this one.
        try:
            backend.delete_collection(palace_path, tmp_name)
        except Exception:
            pass
        curated = client.create_collection(
            tmp_name,
            configuration={"hnsw": {"space": "cosine", "ef_search": ef_search}},
        )

        unreadable = []
        kept = skipped = 0
        for page in _iter_archive(
            archive, ["documents", "metadatas", "embeddings"], unreadable=unreadable
        ):
            ids, docs = page["ids"], page.get("documents") or []
            metas, embs = page.get("metadatas") or [], page.get("embeddings")
            b = ([], [], [], [])
            for i, _id in enumerate(ids):
                meta = metas[i] or {}
                if is_excluded(meta, dumps):
                    skipped += 1
                    continue
                b[0].append(_id)
                b[1].append(docs[i])
                b[2].append(meta)
                b[3].append(embs[i])
            if b[0]:
                curated.add(ids=b[0], documents=b[1], metadatas=b[2], embeddings=b[3])
                kept += len(b[0])
            if progress:
                progress(kept, skipped)

        # Swap only now: the window without a curated collection is milliseconds,
        # instead of the whole rebuild.
        try:
            backend.delete_collection(palace_path, CURATED_COLLECTION)
        except Exception:
            pass
        curated.modify(name=CURATED_COLLECTION)

    if unreadable:
        logger.warning(
            "curated rebuild: %d archive records could not be read and were left "
            "out (offsets %s%s). They come back on the next rebuild once the "
            "vector segment catches up.",
            len(unreadable),
            unreadable[:10],
            " ..." if len(unreadable) > 10 else "",
        )

    return {"kept": kept, "skipped": skipped, "unreadable": len(unreadable)}


def sync(palace_path: str) -> dict:
    """Bring the curated collection up to date with the archive, incrementally.

    Adds drawers mined since the last sync, drops drawers deleted from the
    archive. Cheap enough to run after every mine: the diff is computed from
    id-only scans, and only genuinely new drawers are read with their vectors.

    Returns {"added": n, "removed": n, "skipped": n} — or {"error": ...} if the
    curated collection does not exist yet (run rebuild() first).
    """
    archive = get_collection(palace_path, collection_name=ARCHIVE_COLLECTION, create=False)
    try:
        curated = get_collection(palace_path, collection_name=CURATED_COLLECTION, create=False)
    except Exception:
        return {"error": "no curated collection — run `mempalace curate --rebuild` first"}

    archive_ids, curated_ids = set(), set()
    for page in _iter_archive(archive, [], batch=_ID_BATCH):
        archive_ids.update(page["ids"])
    offset, total = 0, curated.count()
    while offset < total:
        page = curated.get(limit=_ID_BATCH, offset=offset, include=[])
        ids = page.get("ids") or []
        if not ids:
            break
        curated_ids.update(ids)
        offset += len(ids)

    stale = list(curated_ids - archive_ids)
    if stale:
        for i in range(0, len(stale), _BATCH):
            curated.delete(ids=stale[i : i + _BATCH])

    missing = list(archive_ids - curated_ids)

    # Which of the new drawers come from data dumps? A full dump_sources() pass
    # would cost more than the whole sync, so resolve only the source files these
    # drawers actually touch — a mine usually touches a handful.
    candidates = set()
    for i in range(0, len(missing), _BATCH):
        page = archive.get(ids=missing[i : i + _BATCH], include=["metadatas"])
        for meta in page.get("metadatas") or []:
            src = (meta or {}).get("source_file")
            if src and _has_dump_extension(src):
                candidates.add(src)
    dumps = set()
    for src in candidates:
        hits = archive.get(where={"source_file": src}, include=[]).get("ids") or []
        if len(hits) >= DUMP_FANOUT:
            dumps.add(src)

    added = skipped = 0
    for i in range(0, len(missing), _BATCH):
        chunk = missing[i : i + _BATCH]
        page = archive.get(ids=chunk, include=["documents", "metadatas", "embeddings"])
        ids, docs = page.get("ids") or [], page.get("documents") or []
        metas, embs = page.get("metadatas") or [], page.get("embeddings")
        b = ([], [], [], [])
        for j, _id in enumerate(ids):
            meta = metas[j] or {}
            if is_excluded(meta, dumps):
                skipped += 1
                continue
            b[0].append(_id)
            b[1].append(docs[j])
            b[2].append(meta)
            b[3].append(embs[j])
        if b[0]:
            curated.add(ids=b[0], documents=b[1], metadatas=b[2], embeddings=b[3])
            added += len(b[0])

    return {"added": added, "removed": len(stale), "skipped": skipped}


def mirror_upsert(palace_path: str, ids: list) -> None:
    """Mirror specific archive drawers into the curated collection.

    For the single-drawer write paths (MCP add/update), where a full sync would
    be wasteful. Reads the vectors back from the archive rather than
    recomputing, so the two collections can never disagree. Never raises: if
    the mirror fails the drawer is still in the archive, reachable with
    scope="all".
    """
    if not ids:
        return
    try:
        archive = get_collection(palace_path, collection_name=ARCHIVE_COLLECTION, create=False)
        curated = get_collection(palace_path, collection_name=CURATED_COLLECTION, create=False)
        page = archive.get(ids=list(ids), include=["documents", "metadatas", "embeddings"])
        got, docs = page.get("ids") or [], page.get("documents") or []
        metas, embs = page.get("metadatas") or [], page.get("embeddings")
        b = ([], [], [], [])
        for i, _id in enumerate(got):
            meta = metas[i] or {}
            # No dump check here on purpose: this path carries hand-written
            # drawers one at a time, and fan-out cannot be judged from one id
            # without a query on a hot path. The next sync/rebuild settles it.
            if is_history(meta):
                continue
            b[0].append(_id)
            b[1].append(docs[i])
            b[2].append(meta)
            b[3].append(embs[i])
        if b[0]:
            curated.upsert(ids=b[0], documents=b[1], metadatas=b[2], embeddings=b[3])
    except Exception:
        logger.debug("curated mirror_upsert skipped", exc_info=True)


def mirror_delete(palace_path: str, ids: list) -> None:
    """Drop specific drawers from the curated collection. Never raises."""
    if not ids:
        return
    try:
        curated = get_collection(palace_path, collection_name=CURATED_COLLECTION, create=False)
        curated.delete(ids=list(ids))
    except Exception:
        logger.debug("curated mirror_delete skipped", exc_info=True)


def sync_quietly(palace_path: str) -> None:
    """Fire-and-forget sync for the tail of a mine. Never raises."""
    try:
        stats = sync(palace_path)
        if "error" not in stats and (stats["added"] or stats["removed"]):
            logger.info("curated collection synced: +%d -%d", stats["added"], stats["removed"])
    except Exception:
        logger.exception("curated sync failed (search falls back to the full archive)")
