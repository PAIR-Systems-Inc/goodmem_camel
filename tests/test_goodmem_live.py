r"""Live tests for the GoodMem toolkit, against a running GoodMem server.

These skip entirely unless ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL`` are
set, which is also the check that no credential is baked into the package.

Run with::

    GOODMEM_API_KEY=... GOODMEM_BASE_URL=https://localhost:8080 \
        pytest test/toolkits/test_goodmem_live.py -v
"""

import os
import time
import uuid

import pytest

from camel_goodmem import (
    GoodMemError,
    GoodMemRetriever,
    GoodMemToolkit,
    filters,
)
from camel_goodmem._uploads import GoodMemUploadError

API_KEY = os.environ.get("GOODMEM_API_KEY")
BASE_URL = os.environ.get("GOODMEM_BASE_URL")
VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "false").lower() == "true"

pytestmark = pytest.mark.skipif(
    not (API_KEY and BASE_URL),
    reason="GOODMEM_API_KEY and GOODMEM_BASE_URL are not set",
)

RUN = uuid.uuid4().hex[:8]


def _embedder_id(toolkit: GoodMemToolkit) -> str:
    r"""Returns the embedder the live run should use.

    Set ``GOODMEM_TEST_EMBEDDER_ID`` to pin one. Without it the first
    embedder the server lists is used, which is only a guess: an embedder
    whose backing model is unavailable makes every write fail with
    ``EMBEDDER_FAILED``.
    """
    pinned = os.environ.get("GOODMEM_TEST_EMBEDDER_ID")
    if pinned:
        return pinned
    embedders = toolkit.list_embedders()
    assert embedders, "the server has no embedders configured"
    return embedders[0]["embedderId"]


def _second_embedder_id(toolkit: GoodMemToolkit) -> str:
    r"""Returns an embedder that is not the one :func:`_embedder_id` uses."""
    first = _embedder_id(toolkit)
    others = [
        e["embedderId"]
        for e in toolkit.list_embedders()
        if e["embedderId"] != first
    ]
    if not others:
        pytest.skip("need two embedders to test a mismatch")
    return others[0]


@pytest.fixture(scope="module")
def admin() -> GoodMemToolkit:
    r"""A toolkit with the admin surface enabled, used to set the stage."""
    toolkit = GoodMemToolkit(
        verify_ssl=VERIFY_SSL,
        allow_admin_tools=True,
        allow_delete=True,
    )
    yield toolkit
    toolkit.close()


@pytest.fixture(scope="module")
def space(admin: GoodMemToolkit):
    r"""Creates one space for the run and verifies it is gone afterwards."""
    created = admin.create_space(f"camel-live-{RUN}", _embedder_id(admin))
    space_id = created["spaceId"]
    admin.space_ids = [space_id]
    yield space_id

    # Teardown: delete, then confirm against a fresh server inventory rather
    # than trusting the delete call.
    admin._client.spaces.delete(id=space_id)
    remaining = [
        s["name"] for s in admin.list_spaces() if s["spaceId"] == space_id
    ]
    assert not remaining, f"space {space_id} survived teardown"


@pytest.fixture(scope="module")
def seeded(admin: GoodMemToolkit, space: str):
    r"""Stores a memory with an unmistakable identifier and waits for it."""
    canary = f"ORYX-{RUN.upper()}"
    created = admin.goodmem_remember(
        f"The CAMEL live-test canary is {canary}.",
        metadata={"tenant": "acme", "year": 2026, "active": True},
    )
    memory_id = created["memoryId"]
    deadline = time.time() + 60
    while time.time() < deadline:
        if admin.goodmem_search(canary, top_k=3)["totalResults"]:
            break
        time.sleep(2)
    else:
        pytest.fail("the seeded memory never became searchable")
    yield canary, memory_id
    admin.delete_memory(memory_id)


class TestLiveJourney:
    def test_an_exact_identifier_round_trips(self, admin, seeded):
        canary, memory_id = seeded
        result = admin.goodmem_search(canary, top_k=5)
        assert result["partial"] is False
        texts = [hit["text"] for hit in result["results"]]
        assert any(canary in text for text in texts), texts
        assert result["results"][0]["memoryId"] == memory_id

    def test_a_search_never_reaches_a_space_it_was_not_given(
        self, admin, seeded
    ):
        """The negative control is another space, not another query.

        Vector search returns nearest neighbours whatever the query, so
        "search for nonsense and expect nothing" proves nothing. What must
        hold is that content in a space this toolkit was not configured with
        is unreachable.
        """
        other = admin.create_space(
            f"camel-live-other-{RUN}", _embedder_id(admin)
        )
        outsider = GoodMemToolkit(
            space_ids=[other["spaceId"]], verify_ssl=VERIFY_SSL
        )
        secret = f"ADDAX-{uuid.uuid4().hex[:6].upper()}"
        created = outsider.goodmem_remember(f"Other-space canary {secret}.")
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                if outsider.goodmem_search(secret, top_k=3)["totalResults"]:
                    break
                time.sleep(2)
            else:
                pytest.fail("the outsider memory never became searchable")

            # The module's own toolkit is scoped to the run's space only.
            leaked = admin.goodmem_search(secret, top_k=10)
            assert all(
                secret not in hit["text"] for hit in leaked["results"]
            ), "content leaked across spaces"
            assert all(
                hit["spaceId"] != other["spaceId"] for hit in leaked["results"]
            )
        finally:
            outsider.delete_memory(created["memoryId"])
            outsider.close()
            admin._client.spaces.delete(id=other["spaceId"])

    def test_scores_are_higher_is_better_and_keep_the_raw_value(
        self, admin, seeded
    ):
        hit = admin.goodmem_search(seeded[0], top_k=1)["results"][0]
        assert hit["scoreKind"] == "vector"
        assert hit["rawScore"] < 0, "GoodMem vector scores are negative"
        assert hit["score"] > 0, "not flipped to CAMEL's convention"

    def test_memory_metadata_reaches_the_hit(self, admin, seeded):
        hit = admin.goodmem_search(seeded[0], top_k=1)["results"][0]
        assert hit["metadata"]["tenant"] == "acme"
        assert hit["memoryId"] and hit["spaceId"]


class TestLiveStatusContract:
    def test_q4a_a_broken_reranker_returns_hits_and_flags_them(
        self, space, seeded
    ):
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            reranker_id="00000000-0000-7000-8000-000000000000",
        )
        try:
            result = toolkit.goodmem_search(seeded[0], top_k=3)
        finally:
            toolkit.close()
        assert result["totalResults"] > 0, "hits were discarded"
        assert result["partial"] is True
        codes = {s["code"] for s in result["statuses"]}
        assert "RERANKING_FAILED" in codes or "NOT_FOUND" in codes
        assert result["warning"]

    def test_q4b_a_broken_reranker_with_no_hits_returns_empty_and_flags(
        self, admin
    ):
        empty = admin.create_space(
            f"camel-live-empty-{RUN}", _embedder_id(admin)
        )
        toolkit = GoodMemToolkit(
            space_ids=[empty["spaceId"]],
            verify_ssl=VERIFY_SSL,
            reranker_id="00000000-0000-7000-8000-000000000000",
        )
        try:
            result = toolkit.goodmem_search("nothing is stored here", top_k=3)
        finally:
            toolkit.close()
            admin._client.spaces.delete(id=empty["spaceId"])
        assert result["totalResults"] == 0
        assert result["partial"] is True
        assert result["statuses"]

    def test_q1_a_healthy_search_reports_no_status(self, admin, seeded):
        result = admin.goodmem_search(seeded[0], top_k=3)
        assert result["partial"] is False
        assert result["statuses"] == []

    def test_the_read_path_does_not_poll(self, admin):
        empty = admin.create_space(
            f"camel-live-fast-{RUN}", _embedder_id(admin)
        )
        toolkit = GoodMemToolkit(
            space_ids=[empty["spaceId"]], verify_ssl=VERIFY_SSL
        )
        try:
            started = time.time()
            result = toolkit.goodmem_search("nothing at all", top_k=3)
            elapsed = time.time() - started
        finally:
            toolkit.close()
            admin._client.spaces.delete(id=empty["spaceId"])
        assert result["totalResults"] == 0
        assert elapsed < 3.0, f"an empty search took {elapsed:.1f}s"


class TestLiveFilters:
    def test_a_matching_filter_finds_the_memory(self, space, seeded):
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            metadata_filter={"tenant": "acme"},
        )
        try:
            assert toolkit.goodmem_search(seeded[0], top_k=3)["totalResults"]
        finally:
            toolkit.close()

    def test_a_non_matching_filter_excludes_it(self, space, seeded):
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            metadata_filter={"tenant": "other-tenant"},
        )
        try:
            assert (
                toolkit.goodmem_search(seeded[0], top_k=3)["totalResults"] == 0
            )
        finally:
            toolkit.close()

    def test_an_injection_payload_matches_nothing(self, space, seeded):
        """The classic payload must be a value, not syntax."""
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            metadata_filter={"tenant": "x' OR '1'='1"},
        )
        try:
            result = toolkit.goodmem_search(seeded[0], top_k=5)
        finally:
            toolkit.close()
        assert result["totalResults"] == 0, "filter injection succeeded"

    def test_a_boolean_filter_is_cast_as_a_boolean(self, space, seeded):
        """Compared as TEXT the server accepts it and matches nothing."""
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            metadata_filter={"active": True},
        )
        try:
            assert toolkit.goodmem_search(seeded[0], top_k=3)["totalResults"]
        finally:
            toolkit.close()

    def test_the_server_accepts_an_escaped_apostrophe(self, space):
        expression = filters.equals("tenant", "o'brien")
        assert r"\'" in expression
        toolkit = GoodMemToolkit(
            space_ids=[space],
            verify_ssl=VERIFY_SSL,
            metadata_filter={"tenant": "o'brien"},
        )
        try:
            # Accepted by the server: a malformed expression is a 400.
            assert (
                toolkit.goodmem_search("anything", top_k=1)["totalResults"]
                == 0
            )
        finally:
            toolkit.close()


class TestLiveContent:
    def test_text_content_comes_back_as_text(self, admin, seeded):
        out = admin.get_memory(seeded[1], include_content=True)
        assert out["contentEncoding"] == "text"
        assert seeded[0] in out["content"]

    def test_binary_content_round_trips_byte_for_byte(
        self, admin, space, tmp_path
    ):
        import base64

        pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF\n"
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(pdf)
        uploader = GoodMemToolkit(
            space_ids=[space], verify_ssl=VERIFY_SSL, upload_dir=tmp_path
        )
        try:
            created = uploader.goodmem_upload_file("sample.pdf")
            out = admin.get_memory(created["memoryId"], include_content=True)
            assert out["contentEncoding"] == "base64"
            assert base64.b64decode(out["content"]) == pdf
        finally:
            admin.delete_memory(created["memoryId"])
            uploader.close()

    def test_a_missing_memory_is_an_error_with_the_servers_reason(self, admin):
        with pytest.raises(GoodMemError) as err:
            admin.get_memory("00000000-0000-7000-8000-000000000000")
        assert err.value.status_code in (400, 404)


class TestLiveUploads:
    def test_a_host_file_outside_the_upload_dir_is_refused(
        self, space, tmp_path
    ):
        toolkit = GoodMemToolkit(
            space_ids=[space], verify_ssl=VERIFY_SSL, upload_dir=tmp_path
        )
        try:
            with pytest.raises(GoodMemUploadError):
                toolkit.goodmem_upload_file("/etc/hostname")
        finally:
            toolkit.close()

    def test_uploads_are_unavailable_without_an_upload_dir(self, space):
        toolkit = GoodMemToolkit(space_ids=[space], verify_ssl=VERIFY_SSL)
        try:
            names = [t.get_function_name() for t in toolkit.get_tools()]
            assert "goodmem_upload_file" not in names
            with pytest.raises(GoodMemUploadError, match="disabled"):
                toolkit.goodmem_upload_file("anything.txt")
        finally:
            toolkit.close()


class TestLiveSpaces:
    def test_reusing_a_name_with_another_embedder_is_refused(
        self, admin, space
    ):
        other = _second_embedder_id(admin)
        with pytest.raises(GoodMemError) as err:
            admin.create_space(f"camel-live-{RUN}", other)
        assert "cannot be changed" in str(err.value)

    def test_reusing_a_name_with_the_same_embedder_returns_it(
        self, admin, space
    ):
        out = admin.create_space(f"camel-live-{RUN}", _embedder_id(admin))
        assert out["reused"] is True
        assert out["spaceId"] == space

    def test_listing_spaces_follows_pagination(self, admin):
        spaces = admin.list_spaces()
        assert len({s["spaceId"] for s in spaces}) == len(spaces)
        assert any(s["name"] == f"camel-live-{RUN}" for s in spaces)

    def test_a_rejected_create_carries_the_servers_message(self, admin):
        with pytest.raises(GoodMemError) as err:
            admin.create_space(f"camel-live-bad-{RUN}", "not-a-uuid")
        assert err.value.status_code == 400
        assert "embedder" in str(err.value).lower()


class TestLivePublishedRegressions:
    def test_update_space_renames_without_public_read(self, admin, space):
        """0.1.0's update_space sent publicRead and got HTTP 400."""
        renamed = f"camel-live-{RUN}-renamed"
        out = admin.update_space(space, name=renamed)
        assert out["success"] is True
        assert admin.goodmem_get_space(space)["name"] == renamed
        admin.update_space(space, name=f"camel-live-{RUN}")

    def test_labels_merge_on_a_space(self, admin, space):
        admin.update_space(space, labels={"env": "audit"})
        assert admin.goodmem_get_space(space)["labels"]["env"] == "audit"

    def test_list_memories_returns_the_seeded_one(self, admin, space, seeded):
        ids = [m["memoryId"] for m in admin.list_memories(space)]
        assert seeded[1] in ids

    def test_delete_space_really_deletes(self, admin):
        temp = admin.create_space(f"camel-live-del-{RUN}", _embedder_id(admin))
        admin.delete_space(temp["spaceId"])
        remaining = [
            s for s in admin.list_spaces() if s["spaceId"] == temp["spaceId"]
        ]
        assert remaining == []

    def test_binary_content_is_a_string_not_bytes(
        self, admin, space, tmp_path
    ):
        """0.1.0 returned raw bytes, which cannot go into a tool result."""
        import json as _json

        pdf = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n"
            b"trailer<</Root 1 0 R>>\n%%EOF\n"
        )
        (tmp_path / "b.pdf").write_bytes(pdf)
        uploader = GoodMemToolkit(
            space_ids=[space], verify_ssl=VERIFY_SSL, upload_dir=tmp_path
        )
        try:
            created = uploader.goodmem_upload_file("b.pdf")
            out = admin.get_memory(created["memoryId"], include_content=True)
            assert isinstance(out["content"], str)
            _json.dumps(out)
        finally:
            admin.delete_memory(created["memoryId"])
            uploader.close()


class TestLiveRetriever:
    def test_the_retriever_returns_camel_rows(self, admin, space, seeded):
        toolkit = GoodMemToolkit(space_ids=[space], verify_ssl=VERIFY_SSL)
        try:
            rows = GoodMemRetriever(toolkit).query(seeded[0], top_k=3)
        finally:
            toolkit.close()
        assert rows and float(rows[0]["similarity score"]) > 0
        assert rows[0]["extra_info"]["goodmem_score_kind"] == "vector"
        assert rows[0]["extra_info"]["goodmem_partial"] is False

    def test_the_retriever_can_store_and_recall(self, admin, space):
        toolkit = GoodMemToolkit(space_ids=[space], verify_ssl=VERIFY_SSL)
        retriever = GoodMemRetriever(toolkit)
        token = f"IBEX-{uuid.uuid4().hex[:6].upper()}"
        created = retriever.process(f"The retriever canary is {token}.")
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                rows = retriever.query(token, top_k=3)
                if any(token in r.get("text", "") for r in rows):
                    break
                time.sleep(2)
            else:
                pytest.fail("the stored memory never became searchable")
        finally:
            admin.delete_memory(created["memoryId"])
            toolkit.close()
