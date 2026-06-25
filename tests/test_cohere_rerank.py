"""Tests for the optional Cohere Rerank step in searcher._hybrid_rank.

The rerank is opt-in (COHERE_RERANK=1 + COHERE_API_KEY) and fallback-safe:
any failure, or being disabled, must return the hybrid order untouched. We
mock urllib so no network call is ever made.
"""

import json
from unittest.mock import patch

from mempalace import searcher
from mempalace.searcher import (
    _anonymize_for_rerank,
    _cohere_rerank,
    _cohere_rerank_enabled,
)


class _FakeResp:
    """Mimics urlopen's context-manager response."""

    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _hits():
    """Three candidates already in (wrong) hybrid order."""
    return [
        {"text": "desayunar con diarrea", "id": "D1"},
        {"text": "Salarios das empregadas domesticas", "id": "D2"},
        {"text": "relatorio de saldos mensais", "id": "D3"},
    ]


class TestAnonymize:
    def test_strips_nib(self):
        out = _anonymize_for_rerank("conta 004500001234567890123 pagar")
        assert "[NIB]" in out
        assert "1234567890123" not in out

    def test_strips_kz_amount(self):
        out = _anonymize_for_rerank("renda de 350.000,00 Kz este mes")
        assert "[VALOR]" in out
        assert "350.000,00" not in out

    def test_leaves_plain_text(self):
        assert _anonymize_for_rerank("salario Suzete") == "salario Suzete"


class TestEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("COHERE_RERANK", raising=False)
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        assert _cohere_rerank_enabled() is False

    def test_flag_without_key_is_off(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        assert _cohere_rerank_enabled() is False

    def test_flag_and_key_enables(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        assert _cohere_rerank_enabled() is True


class TestRerank:
    def test_disabled_is_noop_no_network(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "0")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        hits = _hits()

        def boom(*a, **k):  # urlopen must never be called when disabled
            raise AssertionError("network hit while disabled")

        with patch("urllib.request.urlopen", side_effect=boom):
            out = _cohere_rerank(hits, "salario Suzete")
        assert [h["id"] for h in out] == ["D1", "D2", "D3"]

    def test_reorders_by_cohere_result(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            # Cohere ranks candidate index 1 first, then 2, then 0.
            return _FakeResp(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.98},
                        {"index": 2, "relevance_score": 0.41},
                        {"index": 0, "relevance_score": 0.02},
                    ]
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _cohere_rerank(_hits(), "salario Suzete")

        assert [h["id"] for h in out] == ["D2", "D3", "D1"]
        assert out[0]["cohere_score"] == 0.98
        assert captured["url"] == searcher._COHERE_RERANK_URL
        assert captured["timeout"] == searcher._COHERE_TIMEOUT_S
        auth = {v for k, v in captured["headers"].items() if k.lower() == "authorization"}
        assert "Bearer key-abc" in auth
        assert captured["body"]["model"] == searcher._COHERE_DEFAULT_MODEL

    def test_anonymizes_before_sending(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp({"results": [{"index": 0, "relevance_score": 0.5}]})

        hits = [{"text": "renda 350.000,00 Kz conta 004500001234567890123", "id": "D1"}]
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _cohere_rerank(hits, "renda")

        sent = captured["body"]["documents"][0]
        assert "350.000,00" not in sent
        assert "1234567890123" not in sent
        assert "[VALOR]" in sent and "[NIB]" in sent

    def test_http_error_falls_back(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")

        def fake_urlopen(req, timeout=None):
            raise OSError("connection refused")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _cohere_rerank(_hits(), "salario Suzete")
        assert [h["id"] for h in out] == ["D1", "D2", "D3"]

    def test_empty_results_falls_back(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")

        def fake_urlopen(req, timeout=None):
            return _FakeResp({"results": []})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _cohere_rerank(_hits(), "salario Suzete")
        assert [h["id"] for h in out] == ["D1", "D2", "D3"]

    def test_only_top_n_sent_tail_preserved(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            # Reverse the two sent candidates.
            return _FakeResp(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                }
            )

        hits = [{"text": f"doc {i}", "id": f"D{i}"} for i in range(4)]
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = _cohere_rerank(hits, "q", top_n=2)

        assert len(captured["body"]["documents"]) == 2  # only top_n sent
        # Reranked window first (reversed), then untouched tail D2, D3.
        assert [h["id"] for h in out] == ["D1", "D0", "D2", "D3"]

    def test_empty_list_is_noop(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")
        assert _cohere_rerank([], "q") == []


class TestHybridRankIntegration:
    def test_hybrid_rank_applies_rerank_when_enabled(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "1")
        monkeypatch.setenv("COHERE_API_KEY", "key-abc")

        def fake_urlopen(req, timeout=None):
            # Force candidate index 1 to the top regardless of hybrid score.
            return _FakeResp({"results": [{"index": 1, "relevance_score": 0.99}]})

        hits = [
            {"text": "alpha beta", "distance": 0.1, "id": "A"},
            {"text": "gamma delta", "distance": 0.9, "id": "B"},
        ]
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            out = searcher._hybrid_rank(hits, "gamma")
        assert out[0]["id"] == "B"

    def test_hybrid_rank_unaffected_when_disabled(self, monkeypatch):
        monkeypatch.setenv("COHERE_RERANK", "0")
        hits = [
            {"text": "alpha beta", "distance": 0.1, "id": "A"},
            {"text": "gamma delta", "distance": 0.9, "id": "B"},
        ]

        def boom(*a, **k):
            raise AssertionError("network hit while disabled")

        with patch("urllib.request.urlopen", side_effect=boom):
            out = searcher._hybrid_rank(hits, "alpha")
        # Pure hybrid order: A (closer vector + bm25 match) stays first.
        assert out[0]["id"] == "A"
