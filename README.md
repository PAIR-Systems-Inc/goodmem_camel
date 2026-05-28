# camel-goodmem

[GoodMem](https://goodmem.ai) integration for
[CAMEL](https://github.com/camel-ai/camel).

GoodMem gives AI agents retrieval-augmented generation (RAG) memory. Store
documents in a space and GoodMem chunks, embeds, and indexes them so your
agent can pull back the most relevant passages on any question.

This package exposes the GoodMem API as a CAMEL `BaseToolkit`. Drop
`GoodMemToolkit` into a `ChatAgent` and the agent can store, list, and
retrieve memories alongside its other tools.

## Installation

```bash
pip install camel-goodmem
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import os

os.environ["GOODMEM_BASE_URL"] = "https://localhost:8080"
os.environ["GOODMEM_API_KEY"] = "gm_xxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["GOODMEM_VERIFY_SSL"] = "false"  # self-signed local server

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from camel_goodmem import GoodMemToolkit

toolkit = GoodMemToolkit(verify_ssl=False)

embedder_id = toolkit.goodmem_list_embedders()[0]["embedderId"]
space_id = toolkit.goodmem_create_space(
    name="quickstart", embedder_id=embedder_id
)["spaceId"]

agent = ChatAgent(
    system_message=(
        f"You are an assistant whose long-term memory lives in GoodMem "
        f"space '{space_id}'. Store facts the user shares, and answer "
        "their questions from that space."
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    ),
    tools=toolkit.get_tools(),
)
```

Each tool returns either a Python list or a `dict` with operation-specific
fields. Errors raise the underlying `requests` exception so the agent can
see and recover from them.

## Available operations

| Method | Description |
|---|---|
| `goodmem_list_embedders` | List embedder models available on the server |
| `goodmem_list_spaces` | List all spaces accessible to the API key |
| `goodmem_get_space` | Fetch a space by ID |
| `goodmem_create_space` | Create a space (idempotent by name) |
| `goodmem_update_space` | Update a space's name, labels, or public-read flag |
| `goodmem_delete_space` | Delete a space and all of its memories |
| `goodmem_create_memory` | Store text or a file as a memory |
| `goodmem_list_memories` | List memories in a space, with pagination and filters |
| `goodmem_retrieve_memories` | Semantic retrieval across one or more spaces |
| `goodmem_get_memory` | Fetch a memory by ID, with optional content |
| `goodmem_delete_memory` | Delete a memory |

### Retrieval options

`goodmem_retrieve_memories` accepts the following parameters in addition to
`query`, `space_ids`, and `max_results`:

| Parameter | Type | Description |
|---|---|---|
| `metadata_filter` | str | SQL-style JSONPath filter applied server-side to every space key. Example: `CAST(val('$.category') AS TEXT) = 'feat'` |
| `wait_for_indexing` | bool | Poll for results when none come back on the first call (default `True`) |
| `max_wait_seconds` | float | Polling budget (default `10`) |
| `poll_interval` | float | Seconds between polls (default `2`) |
| `reranker_id` | str | Reranker model to refine result ordering |
| `llm_id` | str | LLM that generates a contextual abstract reply |
| `relevance_threshold` | float | Minimum score (0-1) for inclusion |
| `llm_temperature` | float | Creativity (0-2) for the LLM post-processor |
| `chronological_resort` | bool | Reorder results by creation time |

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server |
| `GOODMEM_API_KEY` | API key sent as `X-API-Key` |
| `GOODMEM_VERIFY_SSL` | Set to `false` to skip TLS verification (default `true`) |

When the env vars are set, the toolkit constructor can be called with no
arguments.

### Trusting the dev cert on localhost

GoodMem's local dev server ships with TLS on, using a self-signed
certificate. Two ways to handle that during development:

1. Pass `verify_ssl=False` to `GoodMemToolkit(...)`. The toolkit suppresses
   the matching urllib3 `InsecureRequestWarning` so your console stays clean.
2. Set `GOODMEM_VERIFY_SSL=false` in your environment.

For production, install the GoodMem CA certificate into your trust store
and leave `verify_ssl=True`.

## End-to-end example

[`examples/example_usage.py`](examples/example_usage.py) drives the toolkit
through four `ChatAgent` scenarios: persistent project context across
`agent.reset()`, a scribe-and-analyst team pipeline, metadata-driven
retrieval with a server-side filter, and tool-call inspection of the
analyst's response. The answering step uses OpenAI; install with
`pip install camel-goodmem[examples]` and set `OPENAI_API_KEY` before
running.

```bash
python examples/example_usage.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
