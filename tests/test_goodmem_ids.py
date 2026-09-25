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

from camel_goodmem import GoodMemRetriever, GoodMemToolkit

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


def _answer(method: str, path: str) -> tuple[int, str, bytes]:
    r"""Answers like GoodMem would, so a request that gets through succeeds.

    Every DELETE succeeds, which is exactly what made the traversal
    dangerous: the caller was told the memory was deleted.
    """
    as_json = "application/json"
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
            self.command, self.path.split("?", 1)[0]
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

    from camel_goodmem import GoodMemIdError

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
    from camel_goodmem import UUID_PATTERN

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
        from camel_goodmem._ids import require_uuid

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
        from camel_goodmem import GoodMemIdError
        from camel_goodmem._ids import require_uuid

        with pytest.raises(GoodMemIdError, match=r"space_id must be a UUID"):
            require_uuid(value, "space_id")

    def test_the_error_is_a_value_error_like_the_other_input_errors(self):
        from camel_goodmem import GoodMemIdError, GoodMemUploadError

        assert issubclass(GoodMemIdError, ValueError)
        assert issubclass(GoodMemUploadError, ValueError)
