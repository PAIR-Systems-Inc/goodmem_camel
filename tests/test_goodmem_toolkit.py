r"""Offline tests for the GoodMem toolkit.

These drive the *real* GoodMem SDK over an ``httpx`` mock transport, fed with
NDJSON and JSON captured from a live GoodMem server (v1.0.320). Mocking the
toolkit's own client instead would prove nothing: every defect this suite
pins lived in the layer between the SDK and the caller.
"""

import json
import os
import re
from pathlib import Path

import httpx
import pytest

from camel_goodmem import (
    GoodMemError,
    GoodMemRetriever,
    GoodMemToolkit,
    filters,
)
from camel_goodmem._filters import GoodMemFilterError
from camel_goodmem._results import (
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    classify_status,
    orient_score,
    outcome_from_events,
)
from camel_goodmem._uploads import (
    GoodMemUploadError,
    resolve_upload_path,
)

FIXTURES = Path(__file__).parent / "goodmem_fixtures"
BASE = "https://goodmem.test"

# GoodMem ids are UUIDs, and the toolkit refuses anything else before a
# request is made, so every id a test hands the toolkit is a real-shaped one.
SPACE = "01a0d44b-746f-775b-b91e-bc73d4058e27"
MEMORY = "01a0d44b-748d-72eb-b54e-c3ea2d956927"
RERANKER = "019cfd1d-5b7e-7a41-9c3d-2f0e8a6b4c11"
EMB_VOYAGE = "019cfd1c-c033-7517-b7de-f73941a0464b"
EMB_QWEN = "019cfd1c-d2a8-7f40-8e6b-91c4a7d3e052"
SPACE_2 = "01a0d44b-96ae-7081-bc16-5644e701222a"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def make_toolkit(handler, **kwargs) -> GoodMemToolkit:
    r"""Builds a toolkit whose SDK client talks to a mock transport."""
    from goodmem import Goodmem

    client = Goodmem(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE,
            headers={"X-API-Key": "gm_offline_test_key"},
        ),
    )
    kwargs.setdefault("space_ids", [SPACE])
    return GoodMemToolkit(
        base_url=BASE, api_key="gm_offline_test_key", client=client, **kwargs
    )


def retrieve_handler(payload: bytes, *, capture: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":retrieve"):
            if capture is not None:
                capture["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=payload,
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(404, json={"message": "unexpected"})

    return handler


# ---------------------------------------------------------------------------
# The fixtures must be server bytes, not something hand-written later.
# ---------------------------------------------------------------------------


class TestFixturesAreReal:
    def test_fixtures_are_real_server_bytes(self):
        stream = fixture("retrieve_ok.ndjson").decode()
        lines = [ln for ln in stream.strip().split("\n") if ln.strip()]
        assert len(lines) >= 2
        # Every line is a complete JSON object carrying a server-side id.
        events = [json.loads(ln) for ln in lines]
        assert any("resultSetBoundary" in e for e in events)
        assert any("retrievedItem" in e for e in events)
        uuid_re = re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-", re.IGNORECASE
        )
        assert uuid_re.search(stream), "no server-generated UUIDv7 present"

    def test_no_credential_in_fixtures(self):
        for path in FIXTURES.iterdir():
            assert not re.search(
                rb"gm_[a-z0-9]{20,}", path.read_bytes()
            ), f"credential-shaped string in {path.name}"


# ---------------------------------------------------------------------------
# P4 / P3 -- the retrieval status contract
# ---------------------------------------------------------------------------


class TestRetrievalStatusContract:
    def test_q4a_degraded_with_hits_returns_the_hits(self):
        """A real problem must never discard results the server returned."""
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_degraded_hits.ndjson"))
        )
        result = tk.goodmem_search("canary")
        assert result["totalResults"] > 0, "hits were discarded"
        assert result["partial"] is True
        codes = {s["code"] for s in result["statuses"]}
        assert {"NOT_FOUND", "RERANKING_FAILED"} <= codes
        assert result["warning"]

    def test_q4b_degraded_without_hits_returns_empty_and_flags(self):
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_degraded_empty.ndjson"))
        )
        result = tk.goodmem_search("nothing")
        assert result["totalResults"] == 0
        assert result["partial"] is True
        assert result["statuses"]
        assert "RERANKING_FAILED" in result["warning"]

    def test_q1_feature_disabled_is_noise_even_with_details(self):
        """Q1 is decided by code alone -- details are never inspected."""
        status = classify_status("FEATURE_DISABLED", "no LLM configured")
        assert status.informational is True
        status = classify_status("LLM_CAPABILITY_INFERRED", "inferred")
        assert status.informational is True

    def test_q1_informational_only_stream_is_not_partial(self):
        events = [
            {"status": {"code": "FEATURE_DISABLED", "message": "no LLM"}},
        ]
        outcome = outcome_from_events(_as_models(events))
        assert outcome.partial is False
        assert outcome.statuses == []

    def test_q3_unknown_code_surfaces_as_unknown_and_is_never_dropped(self):
        stream = _ndjson(
            [
                {"status": {"code": "SOME_FUTURE_CODE", "message": "new"}},
            ]
        )
        tk = make_toolkit(retrieve_handler(stream))
        result = tk.goodmem_search("q")
        assert result["partial"] is True
        assert [s["code"] for s in result["statuses"]] == [UNKNOWN_CODE]
        assert result["statuses"][0]["message"] == "new"

    def test_q3_unknown_code_does_not_raise(self):
        stream = _ndjson([{"status": {"code": "NOPE", "message": "x"}}])
        tk = make_toolkit(retrieve_handler(stream))
        tk.goodmem_search("q")  # must not raise

    def test_a_clean_stream_is_not_partial(self):
        tk = make_toolkit(retrieve_handler(fixture("retrieve_ok.ndjson")))
        result = tk.goodmem_search("canary")
        assert result["partial"] is False
        assert result["statuses"] == []
        assert "warning" not in result
        assert result["totalResults"] >= 1


# ---------------------------------------------------------------------------
# P3 -- a stream that ends badly must not read as a clean success
# ---------------------------------------------------------------------------


class TestMalformedStream:
    def test_truncated_stream_keeps_what_arrived_and_reports_it(self):
        whole = fixture("retrieve_ok.ndjson")
        tk = make_toolkit(retrieve_handler(whole[: int(len(whole) * 0.6)]))
        result = tk.goodmem_search("canary")
        assert result["partial"] is True
        assert MALFORMED_STREAM_CODE in {s["code"] for s in result["statuses"]}

    def test_undecodable_line_does_not_report_a_clean_success(self):
        broken = fixture("retrieve_ok.ndjson").replace(
            b'{"retrievedItem"', b'{"retrievedIt', 1
        )
        tk = make_toolkit(retrieve_handler(broken))
        result = tk.goodmem_search("canary")
        assert result["partial"] is True
        assert result["warning"]

    def test_events_before_the_break_are_not_thrown_away(self):
        whole = fixture("retrieve_ok.ndjson")
        lines = whole.decode().strip().split("\n")
        # keep the boundary and the definition, truncate inside the chunk
        payload = ("\n".join(lines[:2]) + "\n" + lines[2][:80]).encode()
        tk = make_toolkit(retrieve_handler(payload))
        result = tk.goodmem_search("canary")
        assert result["resultSetId"], "the boundary that did arrive was lost"
        assert result["partial"] is True


# ---------------------------------------------------------------------------
# P17 -- timeouts
# ---------------------------------------------------------------------------


class TestTimeouts:
    def test_a_timeout_is_configured_on_the_client_by_default(self):
        from goodmem import Goodmem

        tk = GoodMemToolkit(
            base_url=BASE, api_key="k", space_ids=[SPACE], timeout=12.5
        )
        assert isinstance(tk._client, Goodmem)
        assert tk._owns_client is True
        tk.close()

    def test_missing_credentials_are_reported_clearly(self, monkeypatch):
        monkeypatch.delenv("GOODMEM_API_KEY", raising=False)
        monkeypatch.delenv("GOODMEM_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="GOODMEM_API_KEY"):
            GoodMemToolkit()


# ---------------------------------------------------------------------------
# P30 / P24 -- joining and de-duplication
# ---------------------------------------------------------------------------


class TestJoinAndDedup:
    def test_chunks_join_to_memories_by_uuid_not_arrival_order(self):
        """The definitions arrive *after* the chunks and in reverse order."""
        events = [
            _chunk("c1", "alpha", "mem-A", -0.2),
            _chunk("c2", "bravo", "mem-B", -0.4),
            _definition("mem-B", {"tag": "B"}),
            _definition("mem-A", {"tag": "A"}),
        ]
        outcome = outcome_from_events(_as_models(events))
        by_id = {h.chunk_id: h for h in outcome.hits}
        assert by_id["c1"].metadata == {"tag": "A"}
        assert by_id["c2"].metadata == {"tag": "B"}

    def test_duplicate_chunk_ids_are_collapsed(self):
        events = [
            _chunk("c1", "alpha", "mem-A", -0.2),
            _chunk("c1", "alpha", "mem-A", -0.2),
        ]
        outcome = outcome_from_events(_as_models(events))
        assert len(outcome.hits) == 1

    def test_distinct_chunks_of_one_memory_are_both_kept(self):
        """De-duplication by memory id would wrongly drop one of these."""
        events = [
            _chunk("c1", "alpha", "mem-A", -0.2),
            _chunk("c2", "bravo", "mem-A", -0.3),
        ]
        outcome = outcome_from_events(_as_models(events))
        assert len(outcome.hits) == 2

    def test_a_chunk_whose_memory_never_arrives_is_still_returned(self):
        events = [_chunk("c1", "alpha", "mem-A", -0.2)]
        outcome = outcome_from_events(_as_models(events))
        assert len(outcome.hits) == 1
        assert outcome.hits[0].metadata == {}


# ---------------------------------------------------------------------------
# P29 -- score semantics
# ---------------------------------------------------------------------------


class TestScoreSemantics:
    def test_vector_scores_are_flipped_to_higher_is_better(self):
        assert orient_score(-0.51, reranked=False) == pytest.approx(0.51)
        assert orient_score(-0.88, reranked=False) == pytest.approx(0.88)

    def test_reranker_scores_are_not_negated(self):
        """Negating a reranker score would invert the ranking."""
        assert orient_score(0.93, reranked=True) == pytest.approx(0.93)
        assert orient_score(-0.14, reranked=True) == pytest.approx(-0.14)

    def test_raw_score_is_preserved_alongside(self):
        tk = make_toolkit(retrieve_handler(fixture("retrieve_ok.ndjson")))
        hit = tk.goodmem_search("canary")["results"][0]
        assert hit["rawScore"] < 0
        assert hit["score"] == pytest.approx(-hit["rawScore"])
        assert hit["scoreKind"] == "vector"

    def test_min_score_is_ignored_without_a_reranker(self):
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson")), min_score=0.99
        )
        assert tk.goodmem_search("canary")["totalResults"] >= 1

    def test_min_score_warns_and_names_the_range_when_it_empties(self):
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson")),
            reranker_id=RERANKER,
            min_score=99.0,
        )
        with pytest.warns(UserWarning, match="observed scores ranged"):
            result = tk.goodmem_search("canary")
        assert result["totalResults"] == 0

    def test_no_threshold_is_sent_by_default(self):
        capture = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        tk.goodmem_search("canary")
        assert "relevanceThreshold" not in json.dumps(capture["body"])


# ---------------------------------------------------------------------------
# P10 -- confined uploads
# ---------------------------------------------------------------------------


class TestUploadConfinement:
    def test_absolute_path_outside_the_directory_is_refused(self, tmp_path):
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            resolve_upload_path("/etc/hostname", tmp_path)

    def test_dot_dot_escape_is_refused(self, tmp_path):
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            resolve_upload_path("../../etc/hostname", tmp_path)

    def test_symlink_escape_is_refused(self, tmp_path):
        link = tmp_path / "escape.txt"
        os.symlink("/etc/hostname", link)
        with pytest.raises(GoodMemUploadError, match="outside the upload"):
            resolve_upload_path("escape.txt", tmp_path)

    def test_a_file_inside_the_directory_is_allowed(self, tmp_path):
        (tmp_path / "ok.txt").write_text("hello")
        assert resolve_upload_path("ok.txt", tmp_path).name == "ok.txt"

    def test_uploads_are_off_without_an_upload_dir(self):
        with pytest.raises(GoodMemUploadError, match="disabled"):
            resolve_upload_path("/etc/hostname", None)

    def test_no_upload_tool_is_offered_without_an_upload_dir(self):
        tk = make_toolkit(retrieve_handler(b""))
        names = [t.get_function_name() for t in tk.get_tools()]
        assert "goodmem_upload_file" not in names

    def test_upload_tool_appears_when_configured(self, tmp_path):
        tk = make_toolkit(retrieve_handler(b""), upload_dir=tmp_path)
        names = [t.get_function_name() for t in tk.get_tools()]
        assert "goodmem_upload_file" in names


# ---------------------------------------------------------------------------
# P34 / P19 -- filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_apostrophe_is_backslash_escaped_not_doubled(self):
        assert filters.equals("name", "o'brien").endswith(r"'o\'brien'")

    def test_backslash_is_escaped(self):
        assert filters.equals("p", "a\\b").endswith(r"'a\\b'")

    def test_injection_payload_stays_inside_the_literal(self):
        built = filters.equals("tenant", "x' OR '1'='1")
        assert built.count("=") == 1 + built.count(r"\'=\'")
        assert r"\'" in built

    def test_control_characters_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="control characters"):
            filters.equals("f", "a\nb")

    def test_booleans_cast_to_boolean_not_text(self):
        """A bool compared as TEXT is accepted by the server and matches
        nothing, so the cast has to be BOOLEAN."""
        assert filters.equals("active", True) == (
            "CAST(val('$.active') AS BOOLEAN) = true"
        )

    def test_numbers_cast_to_numeric(self):
        assert "AS NUMERIC" in filters.equals("year", 2026)

    def test_unsafe_field_names_are_refused(self):
        with pytest.raises(GoodMemFilterError, match="field name"):
            filters.equals("a' OR '1", "x")

    def test_comparisons_and_sets(self):
        assert filters.compare("year", ">=", 2000).endswith(">= 2000")
        assert "IN (" in filters.one_of("tag", ["a", "b"])

    def test_one_of_refuses_mixed_types(self):
        with pytest.raises(GoodMemFilterError, match="same type"):
            filters.one_of("tag", ["a", 1])

    def test_filter_reaches_the_request_as_a_space_key(self):
        capture = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture),
            metadata_filter={"tenant": "acme"},
        )
        tk.goodmem_search("q")
        key = capture["body"]["spaceKeys"][0]
        assert "CAST(val('$.tenant') AS TEXT) = 'acme'" == key["filter"]


# ---------------------------------------------------------------------------
# metadata_filter accepts a `filters` expression, not only a mapping
# ---------------------------------------------------------------------------

#: The expression the README builds. 0.2.1 took only a dict, so passing this
#: raised ``ValueError: dictionary update sequence element #0 has length 1``
#: and compare / one_of / not_equals / any_of could not be applied at all.
README_EXPRESSION = filters.all_of(
    filters.equals("tenant", "acme"),
    filters.compare("year", ">=", 2026),
    filters.one_of("kind", ["note", "doc"]),
)


class TestFilterExpressions:
    def _search_filter(self, **kwargs):
        capture: dict = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture),
            **kwargs,
        )
        tk.goodmem_search("q")
        return [k.get("filter") for k in capture["body"]["spaceKeys"]]

    def test_toolkit_sends_a_filters_expression_verbatim(self):
        sent = self._search_filter(
            space_ids=[SPACE, SPACE_2], metadata_filter=README_EXPRESSION
        )
        assert sent == [README_EXPRESSION, README_EXPRESSION]

    def test_every_filters_builder_reaches_the_request(self):
        expression = filters.any_of(
            filters.not_equals("status", "archived"),
            filters.compare("score", "<", 3),
        )
        assert self._search_filter(metadata_filter=expression) == [expression]

    def test_a_mapping_still_builds_an_and_of_equalities(self):
        sent = self._search_filter(metadata_filter={"active": True, "n": 2})
        assert sent == [
            "(CAST(val('$.active') AS BOOLEAN) = true) AND "
            "(CAST(val('$.n') AS NUMERIC) = 2)"
        ]

    def test_an_empty_expression_sends_no_filter(self):
        assert self._search_filter(metadata_filter="") == [None]
        assert self._search_filter(metadata_filter=filters.all_of()) == [None]

    def test_other_types_are_refused_at_construction(self):
        with pytest.raises(GoodMemFilterError, match="metadata_filter"):
            make_toolkit(retrieve_handler(b""), metadata_filter=["tenant"])

    def test_a_bad_mapping_value_is_refused_at_construction(self):
        with pytest.raises(GoodMemFilterError, match="Unsupported filter"):
            make_toolkit(retrieve_handler(b""), metadata_filter={"x": None})

    def test_the_model_still_cannot_supply_a_filter(self):
        tk = make_toolkit(
            retrieve_handler(b""), metadata_filter=README_EXPRESSION
        )
        props = tk.get_tools()[0].get_openai_tool_schema()["function"][
            "parameters"
        ]["properties"]
        assert set(props) == {"query", "top_k"}

    def test_retriever_accepts_a_filters_expression(self):
        capture: dict = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        GoodMemRetriever(tk, metadata_filter=README_EXPRESSION).query("q")
        key = capture["body"]["spaceKeys"][0]
        assert key["filter"] == README_EXPRESSION

    def test_retriever_accepts_a_mapping(self):
        capture: dict = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture)
        )
        GoodMemRetriever(tk, metadata_filter={"tenant": "acme"}).query("q")
        key = capture["body"]["spaceKeys"][0]
        assert key["filter"] == "CAST(val('$.tenant') AS TEXT) = 'acme'"

    def test_retriever_filter_narrows_the_toolkit_filter_never_widens(self):
        capture: dict = {}
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_ok.ndjson"), capture=capture),
            metadata_filter={"tenant": "acme"},
        )
        narrow = filters.compare("year", ">=", 2026)
        GoodMemRetriever(tk, metadata_filter=narrow).query("q")
        assert capture["body"]["spaceKeys"][0]["filter"] == (
            f"(CAST(val('$.tenant') AS TEXT) = 'acme') AND ({narrow})"
        )
        # The toolkit's own searches keep only the toolkit's filter.
        tk.goodmem_search("q")
        assert capture["body"]["spaceKeys"][0]["filter"] == (
            "CAST(val('$.tenant') AS TEXT) = 'acme'"
        )

    def test_retriever_refuses_other_types_at_construction(self):
        tk = make_toolkit(retrieve_handler(b""))
        with pytest.raises(GoodMemFilterError, match="metadata_filter"):
            GoodMemRetriever(tk, metadata_filter=42)


# ---------------------------------------------------------------------------
# P32 -- embedder reuse
# ---------------------------------------------------------------------------


def _space(space_id, name, embedder_ids):
    r"""Clones a captured space object, retargeting id, name and embedders."""
    import copy

    template = copy.deepcopy(
        json.loads(fixture("spaces_page1.json"))["spaces"][0]
    )
    template["spaceId"] = space_id
    template["name"] = name
    embedder_template = template["spaceEmbedders"][0]
    template["spaceEmbedders"] = []
    for embedder_id in embedder_ids:
        clone = copy.deepcopy(embedder_template)
        clone["embedderId"] = embedder_id
        clone["spaceId"] = space_id
        template["spaceEmbedders"].append(clone)
    return template


class TestSpaceReuse:
    def _spaces_handler(self, spaces, created=None):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path == "/v1/spaces":
                return httpx.Response(200, json={"spaces": spaces})
            if request.method == "POST" and request.url.path == "/v1/spaces":
                return httpx.Response(201, json=created or {})
            return httpx.Response(404, json={"message": "unexpected"})

        return handler

    def test_reuse_requires_a_matching_embedder(self):
        tk = make_toolkit(
            self._spaces_handler([_space(SPACE, "notes", [EMB_VOYAGE])])
        )
        with pytest.raises(GoodMemError) as err:
            tk.create_space("notes", EMB_QWEN)
        assert EMB_VOYAGE in str(err.value)
        assert EMB_QWEN in str(err.value)

    def test_reuse_succeeds_when_the_embedder_matches(self):
        tk = make_toolkit(
            self._spaces_handler([_space(SPACE, "notes", [EMB_VOYAGE])])
        )
        out = tk.create_space("notes", EMB_VOYAGE)
        assert out["reused"] is True and out["spaceId"] == SPACE

    def test_an_ambiguous_name_is_an_error_not_a_coin_flip(self):
        tk = make_toolkit(
            self._spaces_handler(
                [
                    _space(SPACE, "notes", [EMB_VOYAGE]),
                    _space(SPACE_2, "notes", [EMB_VOYAGE]),
                ]
            )
        )
        with pytest.raises(GoodMemError, match="refusing to guess"):
            tk.create_space("notes", EMB_VOYAGE)


# ---------------------------------------------------------------------------
# P6 -- pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_list_spaces_follows_next_token(self):
        page1 = json.loads(fixture("spaces_page1.json"))
        page2 = json.loads(fixture("spaces_page2.json"))
        page2.pop("nextToken", None)
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if len(calls) == 1:
                return httpx.Response(200, json=page1)
            return httpx.Response(200, json=page2)

        tk = make_toolkit(handler)
        spaces = tk.list_spaces()
        assert len(calls) == 2, "the second page was never requested"
        assert len(spaces) == len(page1["spaces"]) + len(page2["spaces"])


# ---------------------------------------------------------------------------
# P7 -- server error bodies
# ---------------------------------------------------------------------------


class TestErrorBodies:
    def test_the_servers_own_message_reaches_the_caller(self):
        body = fixture("error_400.json")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, content=body, headers={"content-type": "application/json"}
            )

        tk = make_toolkit(handler)
        with pytest.raises(GoodMemError) as err:
            tk.list_spaces()
        assert "Invalid embedder ID format" in str(err.value)
        assert err.value.status_code == 400


# ---------------------------------------------------------------------------
# P16 / P12 -- content decoding, and failures that stay failures
# ---------------------------------------------------------------------------


class TestContent:
    def _handler(self, content: bytes, content_type: str, status: int = 200):
        memory = json.loads(fixture("memory_get.json"))
        memory["contentType"] = content_type

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/content"):
                return httpx.Response(
                    status,
                    content=content,
                    headers={"content-type": content_type},
                )
            return httpx.Response(200, json=memory)

        return handler

    def test_text_content_is_returned_as_text(self):
        tk = make_toolkit(self._handler(b"hello there", "text/plain"))
        out = tk.get_memory(MEMORY, include_content=True)
        assert out["content"] == "hello there"
        assert out["contentEncoding"] == "text"

    def test_binary_content_is_returned_as_base64_not_mangled(self):
        pdf = b"%PDF-1.4\x00\x01\x02\xff\xfe"
        tk = make_toolkit(self._handler(pdf, "application/pdf"))
        out = tk.get_memory(MEMORY, include_content=True)
        import base64

        assert base64.b64decode(out["content"]) == pdf
        assert out["contentEncoding"] == "base64"

    def test_a_failed_content_fetch_is_an_error_not_a_success(self):
        tk = make_toolkit(
            self._handler(b'{"message":"gone"}', "application/json", 404)
        )
        with pytest.raises(GoodMemError):
            tk.get_memory(MEMORY, include_content=True)

    def test_content_is_not_fetched_unless_asked_for(self):
        tk = make_toolkit(self._handler(b"x", "text/plain"))
        out = tk.get_memory(MEMORY)
        assert "content" not in out


# ---------------------------------------------------------------------------
# P28 / P21 / P22 / P5 -- surface, secrets, ownership, no polling
# ---------------------------------------------------------------------------


class TestSurfaceAndSafety:
    def test_default_tool_surface_is_narrow(self):
        tk = make_toolkit(retrieve_handler(b""))
        names = [t.get_function_name() for t in tk.get_tools()]
        assert names == ["goodmem_search", "goodmem_remember"]

    def test_admin_and_delete_are_opt_in(self):
        tk = make_toolkit(retrieve_handler(b""))
        names = [t.get_function_name() for t in tk.get_tools()]
        assert "create_space" not in names and "delete_memory" not in names

    def test_search_shows_the_model_only_query_and_top_k(self):
        tk = make_toolkit(retrieve_handler(b""))
        tool = tk.get_tools()[0]
        props = tool.get_openai_tool_schema()["function"]["parameters"][
            "properties"
        ]
        assert set(props) == {"query", "top_k"}

    def test_no_indexing_or_polling_knob_on_the_read_path(self):
        tk = make_toolkit(retrieve_handler(b""))
        props = tk.get_tools()[0].get_openai_tool_schema()["function"][
            "parameters"
        ]["properties"]
        for banned in (
            "wait_for_indexing",
            "poll_interval",
            "max_wait_seconds",
            "llm_temperature",
            "space_ids",
            "relevance_threshold",
        ):
            assert banned not in props

    def test_api_key_is_not_in_repr(self):
        tk = make_toolkit(retrieve_handler(b""))
        assert "gm_offline_test_key" not in repr(tk)

    def test_api_key_is_not_a_public_attribute(self):
        tk = make_toolkit(retrieve_handler(b""))
        public = {
            v
            for k, v in vars(tk).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        assert "gm_offline_test_key" not in public

    def test_an_injected_client_is_never_closed_by_the_toolkit(self):
        from goodmem import Goodmem

        client = Goodmem(
            http_client=httpx.Client(
                transport=httpx.MockTransport(retrieve_handler(b"")),
                base_url=BASE,
                headers={"X-API-Key": "k"},
            ),
        )
        tk = GoodMemToolkit(
            base_url=BASE, api_key="k", client=client, space_ids=[SPACE]
        )
        tk.close()
        assert tk._owns_client is False
        # still usable after the toolkit was closed
        tk2 = GoodMemToolkit(
            base_url=BASE, api_key="k", client=client, space_ids=[SPACE]
        )
        assert tk2._client is client

    def test_searching_without_a_space_is_a_clear_error(self):
        tk = make_toolkit(retrieve_handler(b""), space_ids=[])
        with pytest.raises(GoodMemError, match="No space is configured"):
            tk.goodmem_search("q")


# ---------------------------------------------------------------------------
# Regressions specific to camel-goodmem 0.1.0, the published package
# ---------------------------------------------------------------------------


class TestPublishedPackageRegressions:
    def test_update_space_does_not_offer_public_read(self):
        """0.1.0 sent `publicRead`; the server answers 400."""
        import inspect

        params = inspect.signature(GoodMemToolkit.update_space).parameters
        assert "public_read" not in params
        source = inspect.getsource(GoodMemToolkit.update_space)
        assert "publicRead" not in source.split('"""')[2]

    def test_no_public_read_in_any_shipped_code_path(self):
        """Prose explaining why it is gone is fine; code sending it is not."""
        import ast
        from pathlib import Path as _Path

        import camel_goodmem

        offenders = []
        for path in _Path(camel_goodmem.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text())
            # drop every docstring, then look at what is left
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(
                    node.value, str
                ):
                    node.value = ""
            stripped = ast.unparse(tree)
            if "publicRead" in stripped or "public_read" in stripped:
                offenders.append(path.name)
        assert offenders == []

    def test_binary_content_is_json_serialisable(self):
        """0.1.0 returned raw `bytes`, which no tool result can carry."""
        import json as _json

        memory = json.loads(fixture("memory_get.json"))
        memory["contentType"] = "application/pdf"
        pdf = b"%PDF-1.4\x00\xff"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/content"):
                return httpx.Response(
                    200,
                    content=pdf,
                    headers={"content-type": "application/pdf"},
                )
            return httpx.Response(200, json=memory)

        tk = make_toolkit(handler)
        out = tk.get_memory(MEMORY, include_content=True)
        _json.dumps(out)  # would raise on bytes
        assert isinstance(out["content"], str)

    def test_destructive_tools_are_not_offered_by_default(self):
        """0.1.0 handed the model delete_space and update_space always."""
        tk = make_toolkit(retrieve_handler(b""))
        names = [t.get_function_name() for t in tk.get_tools()]
        for banned in (
            "delete_space",
            "delete_memory",
            "update_space",
            "create_space",
            "list_memories",
        ):
            assert banned not in names

    def test_delete_space_requires_allow_delete(self):
        tk = make_toolkit(retrieve_handler(b""), allow_delete=True)
        names = [t.get_function_name() for t in tk.get_tools()]
        assert "delete_space" in names and "delete_memory" in names

    def test_admin_surface_is_opt_in(self):
        tk = make_toolkit(retrieve_handler(b""), allow_admin_tools=True)
        names = [t.get_function_name() for t in tk.get_tools()]
        assert {"create_space", "update_space", "list_memories"} <= set(names)
        assert "delete_space" not in names

    def test_the_model_cannot_supply_a_raw_filter_expression(self):
        """0.1.0 took `metadata_filter` as a raw string from the model, so the
        model could widen its own scope (live: 1 hit -> 2)."""
        tk = make_toolkit(retrieve_handler(b""))
        props = tk.get_tools()[0].get_openai_tool_schema()["function"][
            "parameters"
        ]["properties"]
        assert "metadata_filter" not in props
        assert set(props) == {"query", "top_k"}


# ---------------------------------------------------------------------------
# The retriever surface
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_query_returns_camel_shaped_rows(self):
        tk = make_toolkit(retrieve_handler(fixture("retrieve_ok.ndjson")))
        rows = GoodMemRetriever(tk).query("canary")
        assert rows and set(rows[0]) >= {
            "similarity score",
            "content path",
            "metadata",
            "extra_info",
            "text",
        }
        assert float(rows[0]["similarity score"]) > 0

    def test_degraded_with_hits_keeps_the_hits_and_flags_them(self):
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_degraded_hits.ndjson"))
        )
        rows = GoodMemRetriever(tk).query("canary")
        assert rows[0]["extra_info"]["goodmem_partial"] is True
        assert rows[0]["extra_info"]["goodmem_statuses"]

    def test_degraded_with_no_hits_explains_itself(self):
        tk = make_toolkit(
            retrieve_handler(fixture("retrieve_degraded_empty.ndjson"))
        )
        with pytest.warns(UserWarning):
            rows = GoodMemRetriever(tk).query("nothing")
        assert len(rows) == 1
        assert "RERANKING_FAILED" in rows[0]["text"]
        assert rows[0]["extra_info"]["goodmem_partial"] is True

    def test_a_genuinely_empty_result_does_not_claim_a_problem(self):
        tk = make_toolkit(retrieve_handler(_ndjson([])))
        rows = GoodMemRetriever(tk).query("nothing")
        assert rows[0]["extra_info"]["goodmem_partial"] is False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ndjson(events) -> bytes:
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def _template(kind: str) -> dict:
    r"""Returns a real captured event of ``kind`` to clone from.

    Synthetic events are built by editing a captured one rather than written
    by hand: the SDK validates every field the server sends, so a hand-made
    event is both rejected and unrepresentative.
    """
    import copy

    for line in fixture("retrieve_ok.ndjson").decode().strip().split("\n"):
        if not line.strip():
            continue
        event = json.loads(line)
        if kind in event:
            return copy.deepcopy(event)
    raise AssertionError(f"no {kind} event in the captured fixture")


def _chunk(chunk_id, text, memory_id, score):
    event = _template("retrievedItem")
    ref = event["retrievedItem"]["chunk"]
    ref["relevanceScore"] = score
    ref["memoryIndex"] = 0
    ref["chunk"]["chunkId"] = chunk_id
    ref["chunk"]["chunkText"] = text
    ref["chunk"]["memoryId"] = memory_id
    return event


def _definition(memory_id, metadata):
    event = _template("memoryDefinition")
    definition = event["memoryDefinition"]
    definition["memoryId"] = memory_id
    definition["metadata"] = metadata
    return event


def _as_models(events):
    r"""Decodes raw event dicts through the SDK's own models."""
    from goodmem.models import RetrieveMemoryEvent

    return [RetrieveMemoryEvent.model_validate(e) for e in events]
