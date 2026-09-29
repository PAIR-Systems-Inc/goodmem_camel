"""CAMEL plugin for GoodMem, a memory service for AI agents.

Documents are chunked, embedded and searched server-side. This package wraps
the official ``goodmem`` SDK and exposes it to CAMEL both as a toolkit and as
a :class:`~camel.retrievers.BaseRetriever`.
"""

from goodmem_camel import filters
from goodmem_camel._filters import GoodMemFilterError
from goodmem_camel._ids import UUID_PATTERN, GoodMemIdError
from goodmem_camel._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from goodmem_camel._uploads import GoodMemUploadError
from goodmem_camel.retriever import GoodMemRetriever
from goodmem_camel.toolkit import GoodMemError, GoodMemToolkit

__version__ = "0.4.0"

__all__ = [
    "GoodMemToolkit",
    "GoodMemRetriever",
    "GoodMemError",
    "GoodMemFilterError",
    "GoodMemIdError",
    "GoodMemUploadError",
    "RetrievalHit",
    "RetrievalOutcome",
    "RetrievalStatus",
    "INFORMATIONAL_CODES",
    "MALFORMED_STREAM_CODE",
    "UNKNOWN_CODE",
    "UUID_PATTERN",
    "filters",
    "__version__",
]
