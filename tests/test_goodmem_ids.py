r"""Ids that reach a URL must be canonical UUIDs, checked before any request.

The ``goodmem`` SDK builds paths by interpolating the id raw
(``f"/v1/memories/{id}"``) and ``httpx`` resolves dot segments before it
sends, so ``delete_memory("../spaces/<id>")`` used to send
``DELETE /v1/spaces/<id>`` -- deleting a whole space -- and report
``success: True``.

These tests drive the real toolkit, the real SDK and real ``httpx`` against a
local HTTP server that records every request line it receives. For every
entry point that takes an id, each malformed id must be refused with an error
naming the field, and the server must have recorded nothing at all.
"""

import ast
import asyncio
import json
import re
import threading
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from camel_goodmem import (
    UUID_PATTERN,
    GoodMemIdError,
    GoodMemRetriever,
    GoodMemToolkit,
)
from camel_goodmem._ids import require_uuid
from tests import test_goodmem_live as live

FIXTURES = Path(__file__).parent / "goodmem_fixtures"
KEY = "gm_offline_test_key"

#: The space a traversal would be aimed at.
VICTIM = "0199d1c0-5a1e-7abc-8def-0123456789ab"
#: The space the toolkit is legitimately configured with.
HOME = "01a0d44b-746f-775b-b91e-bc73d4058e27"
MEMORY = "01a0d44b-748d-72eb-b54e-c3ea2d956927"
EMBEDDER = "019cfd1c-c033-7517-b7de-f73941a0464b"
RERANKER = "019cfd1d-5b7e-7a41-9c3d-2f0e8a6b4c11"

#: Every payload is refused. The first five are traversals, the rest are
#: near-misses a lenient check lets through: a leading space, a query or
#: fragment smuggled after a real id, and a trailing newline -- which a
#: ``^...$`` pattern applied with ``re.match`` accepts.
PAYLOADS = {
    "dot-dot": f"../spaces/{VICTIM}",
    "nested-dot-dot": f"a/../../spaces/{VICTIM}",
    "encoded-dot-dot": f"%2e%2e/spaces/{VICTIM}",
    "encoded-slash": f"..%2Fspaces%2F{VICTIM}",
    "uuid-then-dot-dot": f"{VICTIM}/../../spaces/{VICTIM}",
    "empty": "",
    "leading-space": f" {VICTIM}",
    "query": f"{VICTIM}?x=1",
    "fragment": f"{VICTIM}#frag",
    "trailing-newline": f"{VICTIM}\n",
}


# ---------------------------------------------------------------------------
# A local server that records every request it receives
# ---------------------------------------------------------------------------


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _space_json() -> bytes:
    return json.dumps(
        json.loads(_fixture("spaces_page1.json"))["spaces"][1]
    ).encode()


def _answer(method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
    r"""Answers like GoodMem would, so a request that gets through succeeds.

    Every DELETE succeeds, which is exactly what made the traversal
    dangerous: the caller was told the memory was deleted. Creating a space
    with an embedder the server does not have is refused -- the SDK
    documents that every referenced embedder must exist -- and the live
    suite relies on that. The live server's status for it was not captured,
    so 404 is a stand-in; the live test accepts any 4xx.
    """
    as_json = "application/json"
    if method == "POST" and path == "/v1/spaces":
        embedders = json.loads(body or b"{}").get("spaceEmbedders") or []
        unknown = [
            e.get("embedderId")
            for e in embedders
            if e.get("embedderId") != EMBEDDER
        ]
        if unknown:
            reason = {"message": f"Embedder not found: {unknown[0]}"}
            return 404, as_json, json.dumps(reason).encode()
    if method == "DELETE":
        return 204, as_json, b""
    if path == "/v1/memories:retrieve":
        return 200, "application/x-ndjson", _fixture("retrieve_ok.ndjson")
    if path == "/v1/memories" and method == "POST":
        return 200, as_json, _fixture("memory_get.json")
    if path.startswith("/v1/memories/") and path.endswith("/content"):
        return 200, "text/plain", _fixture("memory_content.txt")
    if path.startswith("/v1/memories/"):
        return 200, as_json, _fixture("memory_get.json")
    if path == "/v1/spaces":
        if method == "POST":
            return 200, as_json, _space_json()
        return 200, as_json, b'{"spaces": []}'
    if re.fullmatch(r"/v1/spaces/[^/]+/memories", path):
        return 200, as_json, b'{"memories": []}'
    if path.startswith("/v1/spaces/"):
        return 200, as_json, _space_json()
    return 404, as_json, b'{"message": "unexpected"}'


class _Recorder(BaseHTTPRequestHandler):
    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _serve(self) -> None:
        body = self._read_body()
        # ``self.path`` is the request target exactly as the client sent it,
        # before any server-side decoding or normalisation.
        self.server.log.append(  # type: ignore[attr-defined]
            {"method": self.command, "target": self.path, "body": body}
        )
        status, content_type, payload = _answer(
            self.command, self.path.split("?", 1)[0], body
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def _httpd():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    httpd.log = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def server(_httpd, monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    _httpd.log.clear()
    return _httpd


def _sent(server) -> list[str]:
    return [f"{r['method']} {r['target']}" for r in server.log]


def _paths(server) -> list[str]:
    r"""The request lines without their query strings."""
    return [
        f"{r['method']} {r['target'].split('?', 1)[0]}" for r in server.log
    ]


def _toolkit(server, **kwargs: Any) -> GoodMemToolkit:
    r"""A toolkit that builds its own SDK client, pointed at the recorder."""
    kwargs.setdefault("space_ids", [HOME])
    kwargs.setdefault("allow_admin_tools", True)
    kwargs.setdefault("allow_delete", True)
    return GoodMemToolkit(
        base_url=f"http://127.0.0.1:{server.server_port}",
        api_key=KEY,
        timeout=5.0,
        **kwargs,
    )


def _assert_refused(server, call: Callable[[], Any], field: str) -> None:
    r"""Asserts ``call`` sent nothing, then that it failed naming ``field``."""
    outcome: Any = None
    error: BaseException | None = None
    try:
        outcome = call()
    except Exception as exc:
        error = exc
    # Checked first, so a failure shows what the server actually received.
    assert _sent(server) == [], (
        f"a malformed {field} reached the server as {_sent(server)}; "
        f"the call returned {outcome!r} and raised {error!r}"
    )
    assert error is not None, f"accepted a malformed {field}: {outcome!r}"

    # A FunctionTool re-raises as a plain ValueError; ours is the context.
    root = error if isinstance(error, GoodMemIdError) else error.__context__
    assert isinstance(root, GoodMemIdError), repr(error)
    assert field in str(root) and "UUID" in str(root), str(root)


# ---------------------------------------------------------------------------
# Every id-taking entry point
# ---------------------------------------------------------------------------

#: Developer-facing methods, each taking an id that lands in a URL path --
#: except ``embedder_id``, which goes in a request body and is checked by the
#: same validator for consistency.
DIRECT: dict[str, tuple[str, Callable[[GoodMemToolkit, str], Any]]] = {
    "get_memory": ("memory_id", lambda tk, v: tk.get_memory(v)),
    "get_memory+content": (
        "memory_id",
        lambda tk, v: tk.get_memory(v, include_content=True),
    ),
    "goodmem_get_space": ("space_id", lambda tk, v: tk.goodmem_get_space(v)),
    "update_space": (
        "space_id",
        lambda tk, v: tk.update_space(v, name="renamed"),
    ),
    "delete_space": ("space_id", lambda tk, v: tk.delete_space(v)),
    "list_memories": ("space_id", lambda tk, v: tk.list_memories(v)),
    "delete_memory": ("memory_id", lambda tk, v: tk.delete_memory(v)),
    "create_space": (
        "embedder_id",
        lambda tk, v: tk.create_space("notes", v),
    ),
}

#: The same methods as a model reaches them: through the FunctionTool that
#: ``get_tools()`` hands to a CAMEL agent, and through ``toolkit.mcp``.
MODEL_TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "get_memory": ("memory_id", {"include_content": True}),
    "goodmem_get_space": ("space_id", {}),
    "update_space": ("space_id", {"name": "renamed"}),
    "delete_space": ("space_id", {}),
    "list_memories": ("space_id", {}),
    "delete_memory": ("memory_id", {}),
    "create_space": ("embedder_id", {"name": "notes"}),
}


def _upload(tk: GoodMemToolkit, tmp_path: Path) -> Any:
    (tmp_path / "note.txt").write_text("hello")
    tk.upload_dir = tmp_path.resolve()
    return tk.goodmem_upload_file("note.txt")


#: Every call that consumes a configured space id. ``list_memories()`` with
#: no argument puts it in a URL path; the rest put it in a request body.
USES_CONFIGURED_SPACE: dict[str, Callable[[GoodMemToolkit, Path], Any]] = {
    "list_memories()": lambda tk, _: tk.list_memories(),
    "goodmem_search": lambda tk, _: tk.goodmem_search("q"),
    "goodmem_remember": lambda tk, _: tk.goodmem_remember("t"),
    "goodmem_upload_file": _upload,
    "retriever.query": lambda tk, _: GoodMemRetriever(tk).query("q"),
    "retriever.process": lambda tk, _: GoodMemRetriever(tk).process("t"),
}

USES_CONFIGURED_RERANKER: dict[str, Callable[[GoodMemToolkit], Any]] = {
    "goodmem_search": lambda tk: tk.goodmem_search("q"),
    "retriever.query": lambda tk: GoodMemRetriever(tk).query("q"),
}


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
@pytest.mark.parametrize("entry", DIRECT.keys())
def test_developer_methods_refuse_a_malformed_id(server, entry, payload):
    field, call = DIRECT[entry]
    tk = _toolkit(server)
    _assert_refused(server, lambda: call(tk, payload), field)


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
@pytest.mark.parametrize("tool", MODEL_TOOLS.keys())
def test_model_tools_refuse_a_malformed_id(server, tool, payload):
    field, extra = MODEL_TOOLS[tool]
    tools = {t.get_function_name(): t for t in _toolkit(server).get_tools()}
    _assert_refused(
        server, lambda: tools[tool](**{field: payload, **extra}), field
    )


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
@pytest.mark.parametrize("tool", MODEL_TOOLS.keys())
def test_mcp_tools_refuse_a_malformed_id(server, tool, payload):
    r"""The MCP server ``@MCPServer`` attaches to every toolkit.

    FastMCP validates arguments against the declared UUID pattern and
    refuses first; the check inside each method stands behind it.
    """
    field, extra = MODEL_TOOLS[tool]
    mcp = _toolkit(server).mcp
    outcome: Any = None
    error: BaseException | None = None
    try:
        outcome = asyncio.run(mcp.call_tool(tool, {field: payload, **extra}))
    except Exception as exc:
        error = exc
    assert _sent(server) == [], (
        f"a malformed {field} reached the server as {_sent(server)}; "
        f"the call returned {outcome!r} and raised {error!r}"
    )
    assert error is not None, f"accepted a malformed {field}: {outcome!r}"
    assert field in str(error), str(error)


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
def test_a_malformed_configured_space_is_refused_at_construction(
    server, payload
):
    _assert_refused(
        server,
        lambda: _toolkit(server, space_ids=[payload]).list_memories(),
        "space_ids",
    )


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
def test_a_malformed_configured_reranker_is_refused_at_construction(
    server, payload
):
    _assert_refused(
        server,
        lambda: _toolkit(server, reranker_id=payload).goodmem_search("q"),
        "reranker_id",
    )


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
@pytest.mark.parametrize("entry", USES_CONFIGURED_SPACE.keys())
def test_a_space_id_set_after_construction_is_refused_at_the_call(
    server, tmp_path, entry, payload
):
    r"""``space_ids`` is a public attribute, so the construction-time check
    alone is not enough: the check has to sit where the id is used."""
    tk = _toolkit(server)
    tk.space_ids = [payload]
    _assert_refused(
        server,
        lambda: USES_CONFIGURED_SPACE[entry](tk, tmp_path),
        "space_ids",
    )


@pytest.mark.parametrize("payload", PAYLOADS.values(), ids=PAYLOADS.keys())
@pytest.mark.parametrize("entry", USES_CONFIGURED_RERANKER.keys())
def test_a_reranker_id_set_after_construction_is_refused_at_the_call(
    server, entry, payload
):
    tk = _toolkit(server)
    tk.reranker_id = payload
    _assert_refused(
        server, lambda: USES_CONFIGURED_RERANKER[entry](tk), "reranker_id"
    )


# ---------------------------------------------------------------------------
# A valid UUID still reaches exactly the intended resource
# ---------------------------------------------------------------------------

VALID_PATHS: dict[str, tuple[Callable[[GoodMemToolkit], Any], list[str]]] = {
    "get_memory": (
        lambda tk: tk.get_memory(MEMORY),
        [f"GET /v1/memories/{MEMORY}"],
    ),
    "get_memory+content": (
        lambda tk: tk.get_memory(MEMORY, include_content=True),
        [f"GET /v1/memories/{MEMORY}", f"GET /v1/memories/{MEMORY}/content"],
    ),
    "goodmem_get_space": (
        lambda tk: tk.goodmem_get_space(VICTIM),
        [f"GET /v1/spaces/{VICTIM}"],
    ),
    "update_space": (
        lambda tk: tk.update_space(VICTIM, name="renamed"),
        [f"PUT /v1/spaces/{VICTIM}"],
    ),
    "delete_space": (
        lambda tk: tk.delete_space(VICTIM),
        [f"DELETE /v1/spaces/{VICTIM}"],
    ),
    "list_memories": (
        lambda tk: tk.list_memories(VICTIM),
        [f"GET /v1/spaces/{VICTIM}/memories"],
    ),
    "list_memories()": (
        lambda tk: tk.list_memories(),
        [f"GET /v1/spaces/{HOME}/memories"],
    ),
    "delete_memory": (
        lambda tk: tk.delete_memory(MEMORY),
        [f"DELETE /v1/memories/{MEMORY}"],
    ),
    "delete_memory, upper-case": (
        lambda tk: tk.delete_memory(MEMORY.upper()),
        [f"DELETE /v1/memories/{MEMORY}"],
    ),
    "delete_memory, uuid.UUID": (
        lambda tk: tk.delete_memory(uuid.UUID(MEMORY)),
        [f"DELETE /v1/memories/{MEMORY}"],
    ),
}


@pytest.mark.parametrize("entry", VALID_PATHS.keys())
def test_a_valid_uuid_reaches_exactly_the_intended_path(server, entry):
    call, expected = VALID_PATHS[entry]
    call(_toolkit(server))
    assert _paths(server) == expected


def test_a_deleted_memory_is_reported_by_its_canonical_id(server):
    out = _toolkit(server).delete_memory(MEMORY.upper())
    assert out == {"success": True, "memoryId": MEMORY}


def test_a_valid_uuid_passes_the_camel_and_mcp_tool_schemas(server):
    tk = _toolkit(server)
    tools = {t.get_function_name(): t for t in tk.get_tools()}
    tools["delete_memory"](memory_id=MEMORY.upper())
    asyncio.run(tk.mcp.call_tool("delete_space", {"space_id": VICTIM}))
    assert _paths(server) == [
        f"DELETE /v1/memories/{MEMORY}",
        f"DELETE /v1/spaces/{VICTIM}",
    ]


def test_body_ids_are_sent_canonical(server):
    tk = _toolkit(
        server, space_ids=[HOME.upper()], reranker_id=RERANKER.upper()
    )
    assert tk.space_ids == [HOME]
    tk.goodmem_search("q")
    tk.create_space("notes", EMBEDDER.upper())
    retrieve, _list, create = server.log
    assert retrieve["target"] == "/v1/memories:retrieve"
    body = json.loads(retrieve["body"])
    assert [k["spaceId"] for k in body["spaceKeys"]] == [HOME]
    assert RERANKER in json.dumps(body) and RERANKER.upper() not in (
        json.dumps(body)
    )
    assert create["method"] == "POST" and create["target"] == "/v1/spaces"
    embedders = json.loads(create["body"])["spaceEmbedders"]
    assert [e["embedderId"] for e in embedders] == [EMBEDDER]


# ---------------------------------------------------------------------------
# The model is told, too
# ---------------------------------------------------------------------------


def test_every_model_visible_id_is_declared_a_uuid(server):
    tools = _toolkit(server).get_tools()
    declared = {}
    for tool in tools:
        schema = tool.get_openai_tool_schema()["function"]
        for name, prop in schema["parameters"]["properties"].items():
            if name.endswith("_id"):
                branches = prop.get("anyOf", [prop])
                patterns = {b["pattern"] for b in branches if "pattern" in b}
                declared[f"{schema['name']}.{name}"] = patterns
    assert declared == {
        f"{tool}.{field}": {UUID_PATTERN}
        for tool, (field, _extra) in MODEL_TOOLS.items()
    }


# ---------------------------------------------------------------------------
# The validator itself
# ---------------------------------------------------------------------------


class TestRequireUuid:
    def test_canonical_ids_are_normalised_to_lower_case(self):
        assert require_uuid(MEMORY, "memory_id") == MEMORY
        assert require_uuid(MEMORY.upper(), "memory_id") == MEMORY
        assert require_uuid(uuid.UUID(MEMORY), "memory_id") == MEMORY

    @pytest.mark.parametrize(
        "value",
        [
            None,
            123,
            b"01a0d44b-748d-72eb-b54e-c3ea2d956927",
            MEMORY.replace("-", ""),
            "{" + MEMORY + "}",
            "urn:uuid:" + MEMORY,
            MEMORY + " ",
            MEMORY[:-1] + "g",
            # Arabic-Indic digit zero: \d would match it, [0-9] does not.
            chr(0x0660) + MEMORY[1:],
        ],
    )
    def test_anything_else_is_refused_naming_the_field(self, value):
        with pytest.raises(GoodMemIdError, match=r"space_id must be a UUID"):
            require_uuid(value, "space_id")

    def test_the_error_is_a_value_error_like_the_other_input_errors(self):
        from camel_goodmem import GoodMemUploadError

        assert issubclass(GoodMemIdError, ValueError)
        assert issubclass(GoodMemUploadError, ValueError)


# ---------------------------------------------------------------------------
# The id sent is the text that was checked, not the caller's object
# ---------------------------------------------------------------------------

_SWAP = f"../spaces/{VICTIM}"


class _SwappingStr(str):
    r"""Holds a real UUID, then turns into a traversal once it is used.

    ``lower()`` is what the validator used to return, and ``__format__`` is
    what the SDK's ``f"/v1/memories/{id}"`` calls.
    """

    def lower(self) -> str:
        return _SWAP

    def __str__(self) -> str:
        return _SWAP

    def __format__(self, spec: str) -> str:
        return _SWAP


class _SwappingUuid(uuid.UUID):
    r"""A UUID whose string form is a traversal."""

    def __str__(self) -> str:
        return _SWAP


def _list_after_setting_space_ids(tk: GoodMemToolkit, value: Any) -> Any:
    tk.space_ids = [value]
    return tk.list_memories()


#: Each id-in-path entry point, the real UUID the swapping value holds, and
#: exactly what must reach the server.
SWAPPABLE: dict[
    str, tuple[Callable[[GoodMemToolkit, Any], Any], str, list]
] = {
    "get_memory+content": (
        lambda tk, v: tk.get_memory(v, include_content=True),
        MEMORY,
        [f"GET /v1/memories/{MEMORY}", f"GET /v1/memories/{MEMORY}/content"],
    ),
    "goodmem_get_space": (
        lambda tk, v: tk.goodmem_get_space(v),
        HOME,
        [f"GET /v1/spaces/{HOME}"],
    ),
    "update_space": (
        lambda tk, v: tk.update_space(v, name="renamed"),
        HOME,
        [f"PUT /v1/spaces/{HOME}"],
    ),
    "delete_space": (
        lambda tk, v: tk.delete_space(v),
        HOME,
        [f"DELETE /v1/spaces/{HOME}"],
    ),
    "list_memories": (
        lambda tk, v: tk.list_memories(v),
        HOME,
        [f"GET /v1/spaces/{HOME}/memories"],
    ),
    "list_memories() with space_ids set later": (
        _list_after_setting_space_ids,
        HOME,
        [f"GET /v1/spaces/{HOME}/memories"],
    ),
    "delete_memory": (
        lambda tk, v: tk.delete_memory(v),
        MEMORY,
        [f"DELETE /v1/memories/{MEMORY}"],
    ),
}


@pytest.mark.parametrize("entry", SWAPPABLE.keys())
def test_a_str_subclass_cannot_swap_the_id_after_the_check(server, entry):
    call, real, expected = SWAPPABLE[entry]
    call(_toolkit(server), _SwappingStr(real.upper()))
    assert _paths(server) == expected


@pytest.mark.parametrize("entry", SWAPPABLE.keys())
def test_a_uuid_subclass_whose_text_is_not_a_uuid_is_refused(server, entry):
    call, real, _expected = SWAPPABLE[entry]
    field = (
        "space_ids" if "space_ids" in entry else DIRECT[entry.split()[0]][0]
    )
    tk = _toolkit(server)
    _assert_refused(server, lambda: call(tk, _SwappingUuid(real)), field)


def test_a_deleted_memory_is_reported_as_a_plain_str(server):
    out = _toolkit(server).delete_memory(_SwappingStr(MEMORY))
    assert type(out["memoryId"]) is str and out["memoryId"] == MEMORY


class TestRequireUuidReturnsPlainText:
    @pytest.mark.parametrize(
        "value",
        [
            MEMORY,
            MEMORY.upper(),
            uuid.UUID(MEMORY),
            _SwappingStr(MEMORY),
            _SwappingStr(MEMORY.upper()),
        ],
        ids=["str", "upper", "uuid.UUID", "str subclass", "upper subclass"],
    )
    def test_the_result_is_a_plain_str_of_the_checked_text(self, value):
        out = require_uuid(value, "memory_id")
        assert type(out) is str
        assert out == MEMORY and f"{out}" == MEMORY and out.lower() == MEMORY

    def test_a_uuid_subclass_is_judged_by_the_text_it_produces(self):
        with pytest.raises(GoodMemIdError, match="memory_id must be a UUID"):
            require_uuid(_SwappingUuid(MEMORY), "memory_id")


# ---------------------------------------------------------------------------
# reranker_id="" meant "no reranker" in 0.2.0; the refusal says what to do
# ---------------------------------------------------------------------------


def test_an_empty_reranker_id_is_refused_saying_how_to_turn_it_off(server):
    _assert_refused(
        server, lambda: _toolkit(server, reranker_id=""), "reranker_id"
    )
    with pytest.raises(GoodMemIdError, match=r"pass reranker_id=None"):
        _toolkit(server, reranker_id="")


def test_an_empty_reranker_id_set_later_says_how_to_turn_it_off(server):
    tk = _toolkit(server)
    tk.reranker_id = ""
    with pytest.raises(GoodMemIdError, match=r"pass reranker_id=None"):
        tk.goodmem_search("q")
    assert _sent(server) == []


def test_no_reranker_is_none_not_an_empty_string(server):
    _toolkit(server, reranker_id=None).goodmem_search("q")
    (retrieve,) = server.log
    assert "rerankerId" not in json.dumps(json.loads(retrieve["body"]))


def test_the_readme_calls_out_that_an_empty_reranker_id_is_refused():
    readme = (Path(__file__).parents[1] / "README.md").read_text("utf-8")
    changes = readme.split("## Changes in 0.2.1", 1)[1].split("\n## ", 1)[0]
    assert 'reranker_id=""' in changes and "reranker_id=None" in changes


# ---------------------------------------------------------------------------
# The live suite must not expect the server to see a malformed id
# ---------------------------------------------------------------------------
# The live tests skip without credentials, so CI never runs them, and one
# that hands the toolkit a malformed id -- expecting the *server* to reject
# it -- fails only on the day someone runs it live. The tests below run the
# live tests that meet the validator against the recorder, and check the
# live source for any other such id.


def test_the_live_rejected_create_holds_against_a_server(server):
    spaces = live.TestLiveSpaces()
    spaces.test_a_rejected_create_carries_the_servers_message(_toolkit(server))
    assert _paths(server) == ["GET /v1/spaces", "POST /v1/spaces"]
    create = json.loads(server.log[1]["body"])
    assert create["spaceEmbedders"][0]["embedderId"] == live.NO_SUCH_EMBEDDER


def test_the_live_malformed_embedder_test_holds_against_a_server(server):
    spaces = live.TestLiveSpaces()
    spaces.test_a_malformed_embedder_id_is_refused_before_it_is_sent(
        _toolkit(server)
    )
    # Only the listing that checks nothing was created.
    assert _paths(server) == ["GET /v1/spaces"]


#: Where a literal id lands in a toolkit call: the positional index of the
#: id for each method, and the keyword and attribute names that carry one.
_ID_POSITION = {
    "get_memory": 0,
    "goodmem_get_space": 0,
    "update_space": 0,
    "delete_space": 0,
    "list_memories": 0,
    "delete_memory": 0,
    "create_space": 1,
}
_ID_NAMES = {"memory_id", "space_id", "embedder_id", "reranker_id", "id"}
_ID_LISTS = {"space_ids"}


def _malformed_live_ids(source: str) -> list[str]:
    r"""Lists literal ids in ``source`` that the validator would refuse,
    other than those inside ``pytest.raises(GoodMemIdError)``."""
    tree = ast.parse(source)
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def text(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    def expects_refusal(item: ast.withitem) -> bool:
        call = item.context_expr
        return (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "raises"
            and bool(call.args)
            and ast.unparse(call.args[0]).endswith("GoodMemIdError")
        )

    refused: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(
            expects_refusal(i) for i in node.items
        ):
            refused.update(id(n) for stmt in node.body for n in ast.walk(stmt))

    found: list[str] = []
    for node in ast.walk(tree):
        ids: list[ast.AST] = []
        if isinstance(node, ast.Call):
            method = getattr(node.func, "attr", None)
            position = _ID_POSITION.get(method or "")
            if position is not None and len(node.args) > position:
                ids.append(node.args[position])
            for kw in node.keywords:
                if kw.arg in _ID_NAMES:
                    ids.append(kw.value)
                elif kw.arg in _ID_LISTS and isinstance(kw.value, ast.List):
                    ids.extend(kw.value.elts)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "attr", None)
                if name in _ID_NAMES:
                    ids.append(node.value)
                elif name in _ID_LISTS and isinstance(node.value, ast.List):
                    ids.extend(node.value.elts)
        for value in ids:
            literal = text(value)
            if (
                literal is not None
                and not re.fullmatch(UUID_PATTERN, literal)
                and id(node) not in refused
            ):
                line = getattr(node, "lineno", "?")
                found.append(f"line {line}: {literal!r}")
    return found


def test_no_live_test_expects_the_server_to_see_a_malformed_id():
    source = Path(live.__file__).read_text("utf-8")
    assert _malformed_live_ids(source) == []


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        # What 0.2.1's first cut left in the live suite.
        (
            "with pytest.raises(GoodMemError):\n"
            "    admin.create_space('n', 'not-a-uuid')\n",
            ["line 2: 'not-a-uuid'"],
        ),
        ("BAD = 'mem-1'\ntk.delete_memory(BAD)\n", ["line 2: 'mem-1'"]),
        ("GoodMemToolkit(space_ids=['space-1'])\n", ["line 1: 'space-1'"]),
        ("GoodMemToolkit(reranker_id='')\n", ["line 1: ''"]),
        ("tk.reranker_id = 'rr-1'\n", ["line 1: 'rr-1'"]),
        # Expected to be refused client-side: fine.
        (
            "with pytest.raises(GoodMemIdError):\n"
            "    admin.create_space('n', 'not-a-uuid')\n",
            [],
        ),
        (f"tk.delete_memory({MEMORY!r})\n", []),
    ],
)
def test_the_live_source_check_finds_a_malformed_id(source, flagged):
    assert _malformed_live_ids(source) == flagged
