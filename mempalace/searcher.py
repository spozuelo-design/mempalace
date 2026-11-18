#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Hybrid search: BM25 keyword matching + vector semantic similarity. The
drawer query is the floor — always runs — and closet hits add a rank-based
boost when they agree. Closets are a ranking *signal*, never a gate, so
weak closets (regex extraction on narrative content) can only help, never
hide drawers the direct path would have found.
"""

import logging
import math
import re
from pathlib import Path

from .palace import get_closets_collection, get_collection

# Closet pointer line format: "topic|entities|→drawer_id_a,drawer_id_b"
# Multiple lines may join with newlines inside one closet document.
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")

logger = logging.getLogger("mempalace_mcp")

# ---------------------------------------------------------------------------
# Which collection do we search?
#
# `mempalace_drawers` holds EVERYTHING, including raw session transcripts. On a
# palace where transcripts dominate (measured here: 111k of 138k drawers, 80%),
# they win every query — they're semantically close to any technical question,
# so curated notes never surface. Filtering them out with a `where` clause does
# NOT work: ChromaDB 1.5.9 post-filters (it finds the HNSW neighbours first,
# then intersects with the metadata filter), so a filtered query returns
# whatever survives — often nothing. Measured: `wing=claude`, 5.580 live
# drawers, returned 0 hits.
#
# So the split is physical: `mempalace_curated` is a second collection holding
# only the non-transcript drawers. Searching it is unfiltered, so HNSW recall
# is intact. `mempalace_drawers` stays untouched as the full archive and is
# still reachable with scope="all".
#
# Fallback-safe: if the curated collection doesn't exist (fresh palace, another
# machine), everything degrades to the full collection exactly as before.
CURATED_COLLECTION = "mempalace_curated"
ARCHIVE_COLLECTION = "mempalace_drawers"

# Candidates fetched from the vector index before any re-ranking, regardless
# of how few results the caller asked for. Feeds the closet-boost stage.
#
# Raised 30 -> 100 on 2026-08-31. See RERANK_POOL below for the measurement.
MIN_RERANK_CANDIDATES = 100
# How many of those the BM25 re-rank is allowed to reorder. See search_memories().
#
# Raised 10 -> 30 on 2026-08-31, measured against the gauntlet gold-set (20
# questions anchored to a known source file) on the Windows palace. The old
# value was tuned on mempalace_bench, a different and much smaller corpus.
#
#   overfetch/pool   hit@1    hit@3         hit@5    noise in top-3
#   30/10 (old)      6/20     9/20 (45%)    9/20     0
#   60/20            8/20    12/20 (60%)   12/20     0
#   100/30           8/20    13/20 (65%)   14/20     0
#   200/50           8/20    12/20 (60%)   14/20     0
#   200/100          9/20    13/20 (65%)   15/20     0
#
# Two things this settles:
#
#  1. The old comment feared a large pool would let BM25 drag up lexical
#     near-misses the vector order had correctly buried. On this gold-set that
#     does not happen: noise in the top-3 stays at ZERO for every size tried.
#     The fear was real on mempalace_bench; it does not generalise.
#  2. The bottleneck was reach, not ranking quality. A separate probe
#     (gauntlet scripts/profundidad_vectorial.py, n=200, no re-ranking) found
#     the correct chunk sitting at vector ranks #12, #23, #28, #35, #55, #86
#     and #102 for questions that were failing. A pool of 10 cannot reach any
#     of them, however good the re-ranker is.
#
# 100/30 over 200/100 deliberately: same hit@3, a third of the BM25 work per
# query, and less room to be fitting 20 questions. The jump from 30/10 to
# anything larger is outside the noise; the choice *among* the larger values
# is not resolved by 20 questions, so do not read 100/30 as an optimum.
#
# Before changing these again, re-run the sweep — it takes two minutes:
#   python <gauntlet>/scripts/barrido_pool.py --k 5
RERANK_POOL = 30
# Closets consulted to build the boost lookup. Fixed on purpose — see the long
# comment in search_memories(). Must be >= RERANK_POOL so the boost can reach
# every candidate the BM25 stage will reorder.
CLOSET_BOOST_CANDIDATES = 30


def resolve_drawers_collection(palace_path: str, scope: str = "curated"):
    """Return (collection, name_used) for the requested search scope.

    scope="curated"  prefer the curated collection, fall back to the archive
    scope="all"      the full archive, transcripts included
    """
    if scope != "all":
        try:
            col = get_collection(palace_path, collection_name=CURATED_COLLECTION, create=False)
            if col.count() > 0:
                return col, CURATED_COLLECTION
        except Exception:
            pass
    return get_collection(palace_path, collection_name=ARCHIVE_COLLECTION, create=False), (
        ARCHIVE_COLLECTION
    )


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _tokenize(text: str) -> list:
    """Lowercase + strip to alphanumeric tokens of length ≥ 2."""
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: list,
    k1: float = 1.5,
    b: float = 0.75,
) -> list:
    """Compute Okapi-BM25 scores for ``query`` against each document.

    IDF is computed over the *provided corpus* using the Lucene/BM25+
    smoothed formula ``log((N - df + 0.5) / (df + 0.5) + 1)``, which is
    always non-negative. This is well-defined for re-ranking a small
    candidate set returned by vector retrieval — IDF then reflects how
    discriminative each query term is *within the candidates*, exactly
    what's needed to reorder them.

    Parameters mirror Okapi-BM25 conventions:
        k1 — term-frequency saturation (1.2-2.0 typical, 1.5 default)
        b  — length normalization (0.0 = none, 1.0 = full, 0.75 default)

    Returns a list of scores in the same order as ``documents``.
    """
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    # Document frequency: how many docs contain each query term?
    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _hybrid_rank(
    results: list,
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list:
    """Re-rank ``results`` by a convex combination of vector similarity and BM25.

    * Vector similarity uses absolute cosine sim ``max(0, 1 - distance)`` —
      ChromaDB's hnsw cosine distance lives in ``[0, 2]`` (0 = identical).
      Absolute (not relative-to-max) means adding/removing a candidate
      can't reshuffle the others.
    * BM25 is real Okapi-BM25 with corpus-relative IDF over the candidates
      themselves. Since the absolute scale is unbounded, BM25 is min-max
      normalized within the candidate set so weights are commensurable.

    Mutates each result dict to add ``bm25_score`` and reorders the list
    in place. Returns the same list for convenience.
    """
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        vec_sim = max(0.0, 1.0 - r.get("distance", 1.0))
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results[:] = [r for _, r in scored]
    return results


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


def _extract_drawer_ids_from_closet(closet_doc: str) -> list:
    """Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.

    Preserves order and dedupes.
    """
    seen: dict = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


def _expand_with_neighbors(drawers_col, matched_doc: str, matched_meta: dict, radius: int = 1):
    """Expand a matched drawer with its ±radius sibling chunks in the same source file.

    Motivation — "drawer-grep context" feature: a closet hit returns one
    drawer, but the chunk boundary may clip mid-thought (e.g., the matched
    chunk says "here's a breakdown:" and the actual breakdown lives in the
    next chunk). Fetching the small neighborhood around the match gives
    callers enough context without forcing a follow-up ``get_drawer`` call.

    Returns a dict with:
        ``text``            combined chunks in chunk_index order
        ``drawer_index``    the matched chunk's index in the source file
        ``total_drawers``   total drawer count for the source file (or None)

    On any ChromaDB failure or missing metadata, falls back to returning the
    matched drawer alone so search never breaks because neighbor expansion
    failed.
    """
    src = matched_meta.get("source_file")
    chunk_idx = matched_meta.get("chunk_index")
    if not src or not isinstance(chunk_idx, int):
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    target_indexes = [chunk_idx + offset for offset in range(-radius, radius + 1)]
    try:
        neighbors = drawers_col.get(
            where={
                "$and": [
                    {"source_file": src},
                    {"chunk_index": {"$in": target_indexes}},
                ]
            },
            include=["documents", "metadatas"],
        )
    except Exception:
        return {"text": matched_doc, "drawer_index": chunk_idx, "total_drawers": None}

    indexed_docs = []
    for doc, meta in zip(neighbors.get("documents") or [], neighbors.get("metadatas") or []):
        ci = meta.get("chunk_index")
        if isinstance(ci, int):
            indexed_docs.append((ci, doc))
    indexed_docs.sort(key=lambda pair: pair[0])

    if not indexed_docs:
        combined_text = matched_doc
    else:
        combined_text = "\n\n".join(doc for _, doc in indexed_docs)

    # Cheap total_drawers lookup: metadata-only scan of the source file.
    total_drawers = None
    try:
        all_meta = drawers_col.get(where={"source_file": src}, include=["metadatas"])
        ids = all_meta.get("ids") or []
        total_drawers = len(ids) if ids else None
    except Exception:
        pass

    return {
        "text": combined_text,
        "drawer_index": chunk_idx,
        "total_drawers": total_drawers,
    }


def search(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    scope: str = "curated",
):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).

    scope="curated" (default) searches curated knowledge only; scope="all"
    includes raw session transcripts. See resolve_drawers_collection().

    Delegates to search_memories() so the CLI and the MCP server rank
    identically. They used to differ — this printed raw vector order while the
    MCP applied closet boosting and BM25 — which meant the CLI quietly returned
    worse results than the same query through Claude. Measured on the
    mempalace_bench suite: raw vector MRR 0.51 against 0.55 for the full path.
    """
    result = search_memories(
        query,
        palace_path,
        wing=wing,
        room=room,
        n_results=n_results,
        scope=scope,
    )

    if "error" in result:
        if result["error"] == "No palace found":
            print(f"\n  No palace found at {palace_path}")
            print("  Run: mempalace init <dir> then mempalace mine <dir>")
        else:
            print(f"\n  {result['error']}")
        raise SearchError(result["error"])

    hits = result["results"]
    if not hits:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    print(f"  Scope: {result['scope']}")
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit['wing']} / {hit['room']}")
        print(f"      Source: {hit['source_file']}")
        print(f"      Match:  {hit['similarity']}")
        print()
        # Print the verbatim text, indented
        for line in hit["text"].strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
    scope: str = "curated",
) -> dict:
    """Programmatic search — returns a dict instead of printing.

    Used by the MCP server and other callers that need data.

    Args:
        query: Natural language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing: Optional wing filter.
        room: Optional room filter.
        n_results: Max results to return.
        max_distance: Max cosine distance threshold. The palace collection uses
            cosine distance (hnsw:space=cosine) — 0 = identical, 2 = opposite.
            Results with distance > this value are filtered out. A value of
            0.0 disables filtering. Typical useful range: 0.3–1.0.
        scope: "curated" (default) searches curated knowledge only; "all"
            includes raw session transcripts. See resolve_drawers_collection().

    Note on wing/room filters: ChromaDB post-filters, so a narrow filter can
    return far fewer results than requested — or none. Prefer plain queries
    against the curated scope over filtering.
    """
    try:
        drawers_col, col_name = resolve_drawers_collection(palace_path, scope)
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    where = build_where_filter(wing, room)

    # Hybrid retrieval: always query drawers directly (the floor), then use
    # closet hits to boost rankings. Closets are a ranking SIGNAL, never a
    # GATE — direct drawer search is always the baseline.
    #
    # This avoids the "weak-closets regression" where narrative content
    # produces low-signal closets (regex extraction matches few topics)
    # and closet-first routing hides drawers that direct search would find.
    # Over-fetch before re-ranking. The multiplier alone is too thin at the
    # small n_results callers actually use: n_results=2 gave the BM25 and
    # closet stages only 6 candidates to work with, so they could not fix a bad
    # vector order. The floor keeps the re-rank pool useful regardless of how
    # many results the caller asked for. Measured on mempalace_bench at k=5:
    # MRR 0.455 -> 0.526, top-1 4/13 -> 5/13, k=10 unchanged. Costs nothing —
    # still a single HNSW query.
    overfetch = max(n_results * 3, MIN_RERANK_CANDIDATES)
    try:
        dkwargs = {
            "query_texts": [query],
            "n_results": overfetch,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            dkwargs["where"] = where
        drawer_results = drawers_col.query(**dkwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

    # Gather closet hits (best-per-source) to build a boost lookup.
    #
    # The size is FIXED, not n_results * 2 as it was until 2026-08-31. Scaling
    # it with the caller's k made the boost table — and therefore the final
    # ordering — depend on how many results you asked for, so asking for MORE
    # could hand back FEWER of the right ones. Measured on the gauntlet
    # gold-set: questions hitting in the top-3 at k=5 vanished entirely at
    # k=50. A caller who widens k to "see more" expects a superset, not a
    # reshuffle.
    #
    # The boost table is a property of the QUERY, not of the page size, so it
    # gets a constant. CLOSET_BOOST_CANDIDATES covers the re-rank pool with
    # room to spare; there is no reason to look at fewer closets just because
    # the caller wants three results instead of thirty.
    closet_boost_by_source: dict = {}  # source_file -> (rank, closet_dist, preview)
    try:
        closets_col = get_closets_collection(palace_path, create=False)
        ckwargs = {
            "query_texts": [query],
            "n_results": CLOSET_BOOST_CANDIDATES,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
        for rank, (cdoc, cmeta, cdist) in enumerate(
            zip(
                closet_results["documents"][0],
                closet_results["metadatas"][0],
                closet_results["distances"][0],
            )
        ):
            source = cmeta.get("source_file", "")
            if source and source not in closet_boost_by_source:
                closet_boost_by_source[source] = (rank, cdist, cdoc[:200])
    except Exception:
        pass  # no closets yet — hybrid degrades to pure drawer search

    # Rank-based boost. The ordinal signal ("which closet matched best") is
    # more reliable than absolute distance on narrative content, where
    # closet distances cluster in 1.2-1.5 range regardless of match quality.
    CLOSET_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]
    CLOSET_DISTANCE_CAP = 1.5  # cosine dist > 1.5 = too weak to use as signal

    scored: list = []
    for doc, meta, dist in zip(
        drawer_results["documents"][0],
        drawer_results["metadatas"][0],
        drawer_results["distances"][0],
    ):
        # Filter on raw distance before rounding to avoid precision loss.
        if max_distance > 0.0 and dist > max_distance:
            continue

        source = meta.get("source_file", "") or ""
        boost = 0.0
        matched_via = "drawer"
        closet_preview = None
        if source in closet_boost_by_source:
            c_rank, c_dist, c_preview = closet_boost_by_source[source]
            if c_dist <= CLOSET_DISTANCE_CAP and c_rank < len(CLOSET_RANK_BOOSTS):
                boost = CLOSET_RANK_BOOSTS[c_rank]
                matched_via = "drawer+closet"
                closet_preview = c_preview

        effective_dist = dist - boost
        entry = {
            "text": doc,
            "wing": meta.get("wing", "unknown"),
            "room": meta.get("room", "unknown"),
            "source_file": Path(source).name if source else "?",
            "similarity": round(max(0.0, 1 - effective_dist), 3),
            "distance": round(dist, 4),
            "effective_distance": round(effective_dist, 4),
            "closet_boost": round(boost, 3),
            "matched_via": matched_via,
            # Internal: retain the full source_file path + chunk_index so the
            # enrichment step below doesn't have to reverse-lookup via
            # basename-suffix matching (which silently collides when two
            # files share a basename across different directories).
            "_sort_key": effective_dist,
            "_source_file_full": source,
            "_chunk_index": meta.get("chunk_index"),
        }
        if closet_preview:
            entry["closet_preview"] = closet_preview
        scored.append(entry)

    scored.sort(key=lambda h: h["_sort_key"])

    # BM25 hybrid re-rank over a BOUNDED pool, then truncate.
    #
    # The size matters in both directions and was measured, not guessed. Too
    # small (the old behaviour: re-rank exactly the n_results kept) and BM25
    # cannot promote the right drawer from rank 8 — at k=3 it only ever
    # shuffled the three it was handed. Too large (the whole 30-candidate
    # over-fetch) and BM25 drags up lexical near-misses the vector order had
    # correctly pushed down. mempalace_bench, MRR (top-3 hits out of 13):
    #
    #     pool         k=3          k=5          k=10
    #     n_results    0.462 (7)    0.526 (9)    0.552 (9)
    #     all 30       0.513 (9)    0.528 (9)    0.528 (9)
    #     max(n,10)    0.538 (10)   0.538 (10)   0.549 (10)
    #
    # The bounded pool wins at every k the callers actually use and is the only
    # one whose behaviour is stable across k. It gives up 0.003 MRR at k=10
    # against re-ranking n_results, and buys a top-3 hit for it.
    #
    # 2026-08-31 — the table above still holds ON mempalace_bench, but its
    # conclusion did NOT generalise. Re-measured on the Windows palace against
    # the gauntlet gold-set, the bounded-at-10 pool was the bottleneck: hit@3
    # 45% at pool 10 vs 65% at pool 30, with noise in the top-3 flat at ZERO for
    # every size tried. RERANK_POOL is now 30. Details and the full sweep are in
    # the RERANK_POOL comment at the top of this module. Do not "restore" 10 on
    # the strength of the mempalace_bench numbers alone — that corpus is much
    # smaller and does not reproduce the effect.
    pool = scored[: max(n_results, RERANK_POOL)]
    _hybrid_rank(pool, query)
    hits = pool[:n_results]

    # Drawer-grep enrichment: for closet-boosted hits whose source has
    # multiple drawers, return the keyword-best chunk + its immediate
    # neighbors instead of just the drawer vector search landed on. The
    # closet said "this source is relevant"; vector may have picked the
    # wrong chunk within it; grep picks the right one.
    MAX_HYDRATION_CHARS = 10000
    for h in hits:
        if h["matched_via"] == "drawer":
            continue
        full_source = h.get("_source_file_full") or ""
        if not full_source:
            continue
        try:
            source_drawers = drawers_col.get(
                where={"source_file": full_source},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue
        docs = source_drawers.get("documents") or []
        metas_ = source_drawers.get("metadatas") or []
        if len(docs) <= 1:
            continue

        # Sort by chunk_index so best_idx + neighbors are positional.
        indexed = []
        for idx, (d, m) in enumerate(zip(docs, metas_)):
            ci = m.get("chunk_index", idx) if isinstance(m, dict) else idx
            if not isinstance(ci, int):
                ci = idx
            indexed.append((ci, d))
        indexed.sort(key=lambda p: p[0])
        ordered_docs = [d for _, d in indexed]

        query_terms = set(_tokenize(query))
        best_idx, best_score = 0, -1
        for idx, d in enumerate(ordered_docs):
            d_lower = d.lower()
            s = sum(1 for t in query_terms if t in d_lower)
            if s > best_score:
                best_score, best_idx = s, idx

        start = max(0, best_idx - 1)
        end = min(len(ordered_docs), best_idx + 2)
        expanded = "\n\n".join(ordered_docs[start:end])
        if len(expanded) > MAX_HYDRATION_CHARS:
            expanded = (
                expanded[:MAX_HYDRATION_CHARS]
                + f"\n\n[...truncated. {len(ordered_docs)} total drawers. "
                "Use mempalace_get_drawer for full content.]"
            )
        h["text"] = expanded
        h["drawer_index"] = best_idx
        h["total_drawers"] = len(ordered_docs)

    for h in hits:
        h.pop("_sort_key", None)
        h.pop("_source_file_full", None)
        h.pop("_chunk_index", None)

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "scope": col_name,
        "total_before_filter": len(drawer_results["documents"][0]),
        "results": hits,
    }
