# camel-goodmem

[GoodMem](https://docs.goodmem.ai) memory for [CAMEL](https://github.com/camel-ai/camel)
agents. Documents are chunked, embedded and searched server-side; this package
wraps the official `goodmem` Python SDK and exposes it to CAMEL both as a
toolkit and as a `BaseRetriever`.

**Version 0.3.0.** Verified against GoodMem server **v1.0.320**.

> **Upgrading from 0.1.0.** 0.1.0 talked to GoodMem over hand-written HTTP and
> had defects that were invisible from its return values — a failed search
> reported `success: true` with no indication anything had gone wrong, and
> `publicRead` was sent on space updates although the server had removed the
> field and answers `400`. See [Changes in 0.2.0](#changes-in-020).

## Install

```bash
pip install camel-goodmem
```

Requires Python 3.10+, `camel-ai>=0.2.79`, `goodmem>=0.1.35`,
`pydantic>=2.11` and `mcp<2`. CI installs exactly those floors and runs the
offline suite against them.

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

Every operational setting — which spaces are readable, which reranker, which
LLM answers from the results, whether a threshold applies, whether files can be
uploaded — is fixed by you at construction time. The model cannot widen its own access, pick another space,
or turn on indexing waits.

Opt in to more:

| Constructor argument | Adds |
| --- | --- |
| `upload_dir=<path>` | `goodmem_upload_file`, confined to that directory |
| `allow_admin_tools=True` | `list_spaces`, `list_embedders`, `goodmem_get_space`, `create_space`, `update_space`, `list_memories`, `get_memory` |
| `allow_delete=True` | `delete_memory`, `delete_space` |
| `allow_write=False` | removes `goodmem_remember` |

### Ids must be UUIDs

Every GoodMem id this package handles — the `space_ids`, `reranker_id` and
`llm_id` you configure, and the `memory_id`, `space_id` and `embedder_id` a tool or method
takes — must be a UUID. Anything else raises `GoodMemIdError`, naming the
argument, **before any request is made**, because the GoodMem SDK puts ids
into request paths unescaped: `delete_memory("../spaces/<id>")` would
otherwise send `DELETE /v1/spaces/<id>` and delete a whole space. Upper-case
UUIDs are accepted and sent lower-case. The tool schemas declare these
arguments with the same UUID pattern, so the model is told up front.

An empty string is not a UUID either: for no reranker, pass
`reranker_id=None` or leave it out. `reranker_id=""` meant "no reranker" in
0.2.0 and is now refused at construction, so
`reranker_id=os.getenv("GOODMEM_RERANKER_ID", "")` fails at startup — write
`os.getenv("GOODMEM_RERANKER_ID") or None`. `llm_id=""` is refused the same
way; for no LLM, pass `llm_id=None` or leave it out.

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
  "abstractReply": "...",     # only with llm_id -- see "LLM answers"
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

So there is **no default threshold**, and `min_score` applies only to
reranker scores. If a threshold removes everything, the toolkit warns and
names the range it actually saw rather than returning a silent empty list.

`scoreKind` says what the server actually did, not what was configured. When
a reranker is set but fails, the server reports `RERANKING_FAILED` (and
`NOT_FOUND` for a missing reranker) and still returns the vector-stage hits.
Those hits are `scoreKind: "vector"`, flipped like any vector score, and
`min_score` is not applied to them, so a reranker threshold cannot discard
them; `partial` is set and `statuses` carries both codes.

## LLM answers

GoodMem can run one of its configured LLMs over the chunks a search retrieved
and return a grounded answer beside them. This is **off by default** and
**set by you**, like `reranker_id`: pass the UUID of a GoodMem LLM as `llm_id`
when you construct the toolkit. The model never sees or chooses it —
`goodmem_search` still takes only `query` and `top_k`.

```python
from camel_goodmem import GoodMemRetriever, GoodMemToolkit

toolkit = GoodMemToolkit(space_ids=["<space-uuid>"], llm_id="<llm-uuid>")

result = toolkit.goodmem_search("What is the canary?")
result["abstractReply"]   # "The canary is **ORYX-2290** ..."
result["results"]         # the hits, exactly as without an LLM

rows = GoodMemRetriever(toolkit).query("What is the canary?")
rows[0]["extra_info"]["goodmem_abstract_reply"]   # the same answer
```

Where the answer appears:

- **`goodmem_search`** (and so the tool result the model reads): the
  `abstractReply` key, a string.
- **`GoodMemRetriever.query()`**: `goodmem_abstract_reply` in every row's
  `extra_info`, since CAMEL's retriever returns a plain list of rows. It is
  one answer for the whole retrieval, repeated on each row so it survives a
  caller keeping only some of them.
- The key is present whenever `llm_id` is set, and absent otherwise.

The id is sent as `llm_id` in the retrieval's post-processor config, beside
`reranker_id` when both are set. It is a UUID like every other id: anything
else raises `GoodMemIdError` before a request is made.

An LLM does not rerank. Hits keep their scores, `scoreKind` and order
exactly as without it; combine it with `reranker_id` if you want reranking
as well.

**When the LLM fails**, the search does not. The server reports
`SUMMARIZATION_FAILED` — plus `NOT_FOUND` when no LLM has that id — and still
returns the hits. You get the hits, `partial: True`, both statuses in
`statuses`, a `warning`, and `abstractReply: None` (in the retriever,
`goodmem_partial: True`, `goodmem_statuses` and `goodmem_abstract_reply:
None`). Nothing is raised and no hit is dropped. Measured live: an LLM id
that does not exist gave `[NOT_FOUND, SUMMARIZATION_FAILED]` with the hit
kept; a provider out of credits gave `SUMMARIZATION_FAILED` carrying the
provider's `429`. A reranker configured beside a failing LLM keeps its
reranker scores.

## Metadata filters

Filters are expressions evaluated server-side, not SQL. You set them when you
construct the toolkit or the retriever; the model never supplies one — in
0.1.0 the filter was a raw string the *model* supplied, which let it widen its
own scope and broke on any value containing an apostrophe.

`metadata_filter` takes either form:

- a **dict** — every pair must match (an `AND` of equalities);
- a **string** built with the `filters` helper — `equals`, `not_equals`,
  `compare`, `one_of`, combined with `all_of` / `any_of` — sent verbatim.

```python
from camel_goodmem import GoodMemRetriever, GoodMemToolkit, filters

# dict: tenant == "acme" AND active == true
toolkit = GoodMemToolkit(
    space_ids=["<space-uuid>"],
    metadata_filter={"tenant": "acme", "active": True},
)

# expression: anything the dict form cannot say
expression = filters.all_of(
    filters.equals("tenant", "acme"),
    filters.compare("year", ">=", 2026),
    filters.one_of("kind", ["note", "doc"]),
)
toolkit = GoodMemToolkit(space_ids=["<space-uuid>"], metadata_filter=expression)
result = toolkit.goodmem_search("quarterly plan")

# the retriever takes the same argument; it is ANDed with the toolkit's
# filter, so a retriever can narrow the toolkit's scope but never widen it
retriever = GoodMemRetriever(
    toolkit,
    metadata_filter=filters.not_equals("status", "archived"),
)
rows = retriever.query("quarterly plan", top_k=5)
```

The helper applies the escaping the server accepts (`'` → `\'`, `\` → `\\`;
SQL-style `''` doubling is rejected with HTTP 400), refuses control characters,
restricts field names, and casts each value to the type GoodMem stored. A
boolean compared as `TEXT` is accepted with HTTP 200 and matches nothing, so
neither `filters` nor the dict form ever stringifies a bool; a `None`, list or
dict value is refused with `GoodMemFilterError` when the toolkit is
constructed. A string is sent as written, so build it with `filters` rather
than by hand.

## Uploads

Uploads are **off** unless you set `upload_dir`. When set, every path is
resolved — symlinks included — and refused if it lands outside that directory,
so a model-supplied path cannot read arbitrary files from the host.

```python
from camel_goodmem import GoodMemToolkit

toolkit = GoodMemToolkit(
    space_ids=["<space-uuid>"], upload_dir="/srv/agent-uploads"
)
```

## Retriever

```python
from camel_goodmem import GoodMemRetriever, GoodMemToolkit

retriever = GoodMemRetriever(GoodMemToolkit(space_ids=["<space-uuid>"]))
retriever.process("Text to remember.")
rows = retriever.query("what did I store?", top_k=5)
```

`query()` returns CAMEL's retriever shape — `similarity score`, `content path`,
`metadata`, `extra_info`, `text` — with GoodMem specifics under `extra_info`
(`goodmem_chunk_id`, `goodmem_memory_id`, `goodmem_space_id`,
`goodmem_score_kind`, `goodmem_raw_score`, `goodmem_partial`,
`goodmem_statuses` when degraded, and `goodmem_abstract_reply` when the
toolkit has an `llm_id`).

## Bringing your own client

```python
from goodmem import Goodmem
from camel_goodmem import GoodMemToolkit

toolkit = GoodMemToolkit(client=Goodmem(base_url=..., api_key=...))
```

An injected client keeps its own server, credentials and TLS settings, and is
never closed by the toolkit.

## Changes in 0.3.0

New, opt-in: an LLM answer from the retrieved chunks. See
[LLM answers](#llm-answers).

| Was (0.2.1) | Now |
| --- | --- |
| No way to ask for GoodMem's LLM post-processing: `GoodMemToolkit(llm_id=...)` raised `TypeError: unexpected keyword argument 'llm_id'`, and no request carried one, so the `abstractReply` the result parser could read never arrived | `llm_id` constructor argument, checked as a UUID before any request and sent in the post-processor config; the answer is `abstractReply` on `goodmem_search` and `goodmem_abstract_reply` in the retriever's `extra_info`. Live with an OpenRouter `qwen/qwen3-8b` LLM: "The fixture canary is **ORYX-2290** ..." |
| Not reachable: no LLM could be requested | A failing LLM keeps the hits: `partial: True`, `statuses` `[NOT_FOUND, SUMMARIZATION_FAILED]` for an id that does not exist, `[SUMMARIZATION_FAILED]` for a provider `429`, `abstractReply: None`, never an exception |
| Not reachable | `goodmem_search(query, top_k)` is unchanged: the model cannot set or see the LLM |

## Changes in 0.2.1

Measured against a local server that records every request line, driving the
real SDK and `httpx`:

| Was (0.2.0) | Now |
| --- | --- |
| `delete_memory("../spaces/<id>")` sent `DELETE /v1/spaces/<id>` and returned `{"success": True}`; the same traversal reached `delete_space`, `update_space`, `goodmem_get_space`, `get_memory` and `list_memories` | Refused with `GoodMemIdError` naming the argument; nothing is sent |
| `%2e%2e/…`, `..%2F…`, a leading space, `?x=1` and `#frag` after an id all reached the server; `list_memories("<id>#frag")` requested a different endpoint, `GET /v1/spaces/<id>` | Only a canonical UUID is accepted |
| `list_memories("")` silently listed the configured space | Refused |
| A malformed `space_ids` or `reranker_id` was sent as-is, and `list_memories()` put the configured space id in a URL path | Refused at construction, and again at every use |
| `reranker_id=""` meant "no reranker" | **Refused** with `GoodMemIdError` at construction; the message says to pass `reranker_id=None` |
| Declared `camel-ai>=0.2.0` and `pydantic>=2`, neither true: camel-ai 0.2.0/0.2.10 fail to import this package (`No module named 'camel.logger'`), 0.2.20/0.2.59 fail on camel-ai's own undeclared `PIL`, and 0.2.60–0.2.78 import it but turn every exception a method raises into `IndexError` (245 of 466 offline tests fail; `delete_memory("../spaces/<id>")` raised `IndexError`, not `GoodMemIdError`). On Python 3.10 pydantic 2.10 made `get_tools()` raise `TypeError` | `camel-ai>=0.2.79`, `pydantic>=2.11` (`mcp<2` kept); a CI `floors` job installs them with `--resolution lowest-direct`, checks the installed versions equal the declared floors, imports the package and runs the offline suite |
| With `reranker_id` set and the reranker failing, the server's vector fallback hits (raw `-0.5846`) were labelled `scoreKind: "reranker"` from configuration and left un-negated (`score: -0.5846`); `min_score=0.0` then removed every hit the server returned | `scoreKind`/orientation come from the response: `RERANKING_FAILED` or a reranker `NOT_FOUND` means `vector`, `score: 0.5846`, `min_score` skipped, the hit kept, `partial: true` with both statuses |
| `metadata_filter` took only a dict, so the expression this README built with `filters` raised `ValueError: dictionary update sequence element #0 has length 1; 2 is required`, and `compare` / `one_of` / `not_equals` / `any_of` could not be applied at all; `GoodMemRetriever` took no filter | `metadata_filter` is `dict` or a `filters` expression string (sent verbatim) on the toolkit and the retriever; the retriever's is ANDed with the toolkit's. A bad filter fails at construction |

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
| `tests/test_goodmem_toolkit.py` | 101 | nothing — the real SDK over a mock transport, fed NDJSON captured from a live server |
| `tests/test_goodmem_ids.py` | 416 | nothing — the real SDK and `httpx` against a local server that records every request; every id-taking entry point (method, CAMEL tool, MCP tool, configuration) × ten malformed ids must send nothing, and a `str` or `uuid.UUID` subclass cannot change the id after it is checked. It also runs the live tests that depend on the id check against that server, and fails if any other live test passes an id the check would refuse |
| `tests/test_goodmem_live.py` | 35 | `GOODMEM_API_KEY` + `GOODMEM_BASE_URL`; skips entirely without them. The LLM tests also take `GOODMEM_TEST_LLM_ID` (a working LLM), and optionally `GOODMEM_TEST_FAILING_LLM_ID` (one whose provider fails) and `GOODMEM_TEST_RERANKER_ID`; each skips without its id |

```bash
pip install -e ".[dev]"

# offline
pytest tests/test_goodmem_toolkit.py tests/test_goodmem_ids.py

# live (pin the embedder if the server's first one is unhealthy)
GOODMEM_API_KEY=... GOODMEM_BASE_URL=... \
  GOODMEM_TEST_EMBEDDER_ID=... GOODMEM_TEST_LLM_ID=... \
  pytest tests/test_goodmem_live.py

# what CI runs
ruff check camel_goodmem tests
ruff format --check camel_goodmem tests
mypy camel_goodmem

# ...and the declared floors, on Python 3.10
uv venv --python 3.10 floor
uv pip install --python floor/bin/python --resolution lowest-direct -e .
uv pip install --python floor/bin/python pytest pytest-timeout
floor/bin/python -m pytest tests/test_goodmem_toolkit.py tests/test_goodmem_ids.py
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
