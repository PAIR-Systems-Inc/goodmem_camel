"""CAMEL plugin for GoodMem, a memory service for AI agents.

Documents are chunked, embedded and searched server-side. This package wraps
the official ``goodmem`` SDK and exposes it to CAMEL both as a toolkit and as
a :class:`~camel.retrievers.BaseRetriever`.
"""

from camel_goodmem import filters
from camel_goodmem._filters import GoodMemFilterError
from camel_goodmem._ids import UUID_PATTERN, GoodMemIdError
from camel_goodmem._results import (
    INFORMATIONAL_CODES,
    MALFORMED_STREAM_CODE,
    UNKNOWN_CODE,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalStatus,
)
from camel_goodmem._uploads import GoodMemUploadError
from camel_goodmem.retriever import GoodMemRetriever
from camel_goodmem.toolkit import GoodMemError, GoodMemToolkit

__version__ = "0.2.1"

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
