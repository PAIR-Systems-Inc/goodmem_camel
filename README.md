# camel-goodmem

[GoodMem](https://docs.goodmem.ai) memory for [CAMEL](https://github.com/camel-ai/camel)
agents. Documents are chunked, embedded and searched server-side; this package
wraps the official `goodmem` Python SDK and exposes it to CAMEL both as a
toolkit and as a `BaseRetriever`.

**Version 0.2.1.** Verified against GoodMem server **v1.0.320**.

> **Upgrading from 0.1.0.** 0.1.0 talked to GoodMem over hand-written HTTP and
> had defects that were invisible from its return values — a failed search
> reported `success: true` with no indication anything had gone wrong, and
> `publicRead` was sent on space updates although the server had removed the
> field and answers `400`. See [Changes in 0.2.0](#changes-in-020).

## Install

```bash
pip install camel-goodmem
```

```bash
export GOODMEM_API_KEY="gm_your_key_here"
export GOODMEM_BASE_URL="https://your-goodmem-server"
```

## Use

```python
from camel.agents import ChatAgent
from camel_goodmem import GoodMemToolkit

toolkit = GoodMemToolkit(space_ids=["<space-uuid>"])
agent = ChatAgent("You remember things.", tools=toolkit.get_tools())
```

By default the model sees exactly two tools:

| Tool | What the model may pass |
| --- | --- |
| `goodmem_search` | `query`, `top_k` |
| `goodmem_remember` | `text`, `metadata` |

Every operational setting — which spaces are readable, which reranker, whether
a threshold applies, whether files can be uploaded — is fixed by you at
construction time. The model cannot widen its own access, pick another space,
or turn on indexing waits.

Opt in to more:

| Constructor argument | Adds |
| --- | --- |
| `upload_dir=<path>` | `goodmem_upload_file`, confined to that directory |
| `allow_admin_tools=True` | `list_spaces`, `list_embedders`, `goodmem_get_space`, `create_space`, `update_space`, `list_memories`, `get_memory` |
| `allow_delete=True` | `delete_memory`, `delete_space` |
| `allow_write=False` | removes `goodmem_remember` |

### Ids must be UUIDs

Every GoodMem id this package handles — the `space_ids` and `reranker_id` you
configure, and the `memory_id`, `space_id` and `embedder_id` a tool or method
takes — must be a UUID. Anything else raises `GoodMemIdError`, naming the
argument, **before any request is made**, because the GoodMem SDK puts ids
into request paths unescaped: `delete_memory("../spaces/<id>")` would
otherwise send `DELETE /v1/spaces/<id>` and delete a whole space. Upper-case
UUIDs are accepted and sent lower-case. The tool schemas declare these
arguments with the same UUID pattern, so the model is told up front.

## Retrieval results

```python
{
  "success": True,
  "query": "...",
  "results": [
    {
      "chunkId": "...", "text": "...", "memoryId": "...", "spaceId": "...",
      "score": 0.64,          # higher is better
      "rawScore": -0.64,      # exactly what the server sent
      "scoreKind": "vector",  # or "reranker" -- not the same scale
      "contentType": "text/plain",
      "metadata": {...},      # the memory's metadata, joined by UUID
    }
  ],
  "totalResults": 1,
  "partial": False,           # True when the server reported a problem
  "statuses": [],             # what it reported
  "resultSetId": "...",
}
```

`partial` means exactly one thing: **the server reported a real problem during
this retrieval.** It is independent of whether hits came back. A degraded
search still returns whatever hits arrived, with `partial` set; when nothing
usable arrives the result is empty, `partial` is set, and a `warning` key plus
a WARNING log line carry the server's own reason. A failed search is never
presented as an empty one.

### Scores

GoodMem produces two kinds of score, and they are not comparable:

- **vector** scores are negative distances. `score` is the flipped value so
  higher is better, with `rawScore` kept beside it.
- **reranker** scores are already higher-is-better, on a **provider-dependent**
  scale. Measured live on the same five documents: Voyage `rerank-2.5` returned
  `0.27..0.93`, Jina `jina-reranker-v3` returned `-0.14..0.43`.

So there is **no default threshold**, and `min_score` applies only when
`reranker_id` is set. If a threshold removes everything, the toolkit warns and
names the range it actually saw rather than returning a silent empty list.

## Metadata filters

Filters are expressions evaluated server-side, not SQL. Build them with the
`filters` helper — in 0.1.0 the filter was a raw string the *model* supplied,
which let it widen its own scope and broke on any value containing an
apostrophe:

```python
from camel_goodmem import GoodMemToolkit, filters

toolkit = GoodMemToolkit(
    space_ids=["..."],
    metadata_filter={"tenant": "acme", "active": True},
)

expression = filters.all_of(
    filters.equals("tenant", "acme"),
    filters.compare("year", ">=", 2026),
    filters.one_of("kind", ["note", "doc"]),
)
```

The helper applies the escaping the server accepts (`'` → `\'`, `\` → `\\`;
SQL-style `''` doubling is rejected with HTTP 400), refuses control characters,
restricts field names, and casts each value to the type GoodMem stored. A
boolean compared as `TEXT` is accepted with HTTP 200 and matches nothing, so
`filters` never stringifies a bool.

## Uploads

Uploads are **off** unless you set `upload_dir`. When set, every path is
resolved — symlinks included — and refused if it lands outside that directory,
so a model-supplied path cannot read arbitrary files from the host.

```python
toolkit = GoodMemToolkit(space_ids=["..."], upload_dir="/srv/agent-uploads")
```

## Retriever

```python
from camel_goodmem import GoodMemRetriever, GoodMemToolkit

retriever = GoodMemRetriever(GoodMemToolkit(space_ids=["..."]))
retriever.process("Text to remember.")
rows = retriever.query("what did I store?", top_k=5)
```

`query()` returns CAMEL's retriever shape — `similarity score`, `content path`,
`metadata`, `extra_info`, `text` — with GoodMem specifics under `extra_info`
(`goodmem_chunk_id`, `goodmem_memory_id`, `goodmem_space_id`,
`goodmem_score_kind`, `goodmem_raw_score`, `goodmem_partial`, and
`goodmem_statuses` when degraded).

## Bringing your own client

```python
from goodmem import Goodmem
from camel_goodmem import GoodMemToolkit

toolkit = GoodMemToolkit(client=Goodmem(base_url=..., api_key=...))
```

An injected client keeps its own server, credentials and TLS settings, and is
never closed by the toolkit.

## Changes in 0.2.1

Measured against a local server that records every request line, driving the
real SDK and `httpx`:

| Was (0.2.0) | Now |
| --- | --- |
| `delete_memory("../spaces/<id>")` sent `DELETE /v1/spaces/<id>` and returned `{"success": True}`; the same traversal reached `delete_space`, `update_space`, `goodmem_get_space`, `get_memory` and `list_memories` | Refused with `GoodMemIdError` naming the argument; nothing is sent |
| `%2e%2e/…`, `..%2F…`, a leading space, `?x=1` and `#frag` after an id all reached the server; `list_memories("<id>#frag")` requested a different endpoint, `GET /v1/spaces/<id>` | Only a canonical UUID is accepted |
| `list_memories("")` silently listed the configured space | Refused |
| A malformed `space_ids` or `reranker_id` was sent as-is, and `list_memories()` put the configured space id in a URL path | Refused at construction, and again at every use |

## Changes in 0.2.0

Every item below was reproduced against the published 0.1.0 wheel, live
against GoodMem v1.0.320.

| Was | Now |
| --- | --- |
| Hand-written `requests` client | Official `goodmem` SDK |
| A search with a broken reranker returned `success: true` and no status; the server had sent three | `partial` + `statuses`, and the hits are still returned |
| `publicRead` sent on space update — live `400 Unrecognized field "publicRead"` | Not offered; the SDK's own request model has no such field |
| `metadata_filter` was a raw string from the model, so it could widen its own scope; an apostrophe in a value was a `400` | Developer-set `metadata_filter`, built and escaped by `filters` |
| `file_path` was a model argument with no restriction; it read `/etc/hostname` and uploaded it | Confined to `upload_dir`; absolute, `..` and symlink escapes refused |
| Empty search took **11.6 s** — `wait_for_indexing` defaulted on and was model-controllable | **0.33 s**; the read path never polls |
| 13 retrieval arguments; `delete_space` and `update_space` always in the toolset | `goodmem_search(query, top_k)`; admin and destructive tools opt-in |
| A PDF's content came back as raw `bytes`, which no tool result can carry | Text as text, anything else base64 — always JSON-serialisable |
| A failed content fetch set `contentError` and left `success: true` | A failure raises |
| Chunks and memories were two arrays joined by position | Joined by UUID, de-duplicated by chunk id |
| Threshold documented "(0-1)"; raw negative scores | `score`/`rawScore`/`scoreKind`, reranker-only threshold that warns |
| `list_spaces` returned the first page; the server's `nextToken` was never read | Paginated, bounded by `max_list_items` |
| `400 Client Error: Bad Request` | The server's own message and status on `GoodMemError` |
| Reusing a space name silently accepted a different embedder | Reuse requires a matching embedder; a mismatch names both |
| No request carried a timeout (0 of 12) | On the client, configurable |
| No retriever — GoodMem could not be used with CAMEL's RAG paths | `GoodMemRetriever(BaseRetriever)` |
| 71 tests that mocked the HTTP session wholesale; no CI | 69 offline + 29 live; CI on 3.10–3.13 |

## Tests

| Suite | Count | Needs |
| --- | --- | --- |
| `tests/test_goodmem_toolkit.py` | 69 | nothing — the real SDK over a mock transport, fed NDJSON captured from a live server |
| `tests/test_goodmem_ids.py` | 345 | nothing — the real SDK and `httpx` against a local server that records every request; every id-taking entry point (method, CAMEL tool, MCP tool, configuration) × ten malformed ids must send nothing |
| `tests/test_goodmem_live.py` | 29 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them |

```bash
pip install -e ".[dev]"

# offline
pytest tests/test_goodmem_toolkit.py tests/test_goodmem_ids.py

# live (pin the embedder if the server's first one is unhealthy)
GOODMEM_API_KEY=... GOODMEM_BASE_URL=... \
  GOODMEM_TEST_EMBEDDER_ID=... \
  pytest tests/test_goodmem_live.py

# what CI runs
ruff check camel_goodmem tests
ruff format --check camel_goodmem tests
mypy camel_goodmem
```

The live suite creates one space per run and asserts, against a fresh server
inventory, that it is gone afterwards.

## Deliberately not done

- **No `AgentMemory` implementation.** CAMEL's `AgentMemory` is chat history
  with a context-window policy; GoodMem is a document store with server-side
  embedding. Implementing it would fake one side of the contract.
- **Turning off TLS verification** is possible via `verify_ssl` for
  self-signed development servers. It defaults to on, no example here turns it
  off, and CI fails if shipped Python does.

## License

Apache-2.0.
