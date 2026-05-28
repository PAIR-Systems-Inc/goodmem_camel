"""CAMEL plugin for GoodMem, the retrieval-augmented generation (RAG) memory backend for AI agents."""

from camel_goodmem.goodmem_toolkit import GoodMemToolkit, _get_mime_type

__all__ = [
    "GoodMemToolkit",
    "_get_mime_type",
]
