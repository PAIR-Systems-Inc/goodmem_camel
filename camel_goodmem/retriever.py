import warnings
from typing import Any

from camel.logger import get_logger
from camel.retrievers.base import BaseRetriever

logger = get_logger(__name__)

DEFAULT_TOP_K_RESULTS = 5


class GoodMemRetriever(BaseRetriever):
    r"""Retrieves from GoodMem through CAMEL's retriever interface.

    This is the surface CAMEL's RAG paths consume, so GoodMem can be used
    wherever a :class:`~camel.retrievers.BaseRetriever` is accepted rather
    than only through tool calls.

    Chunking and embedding happen server-side when a memory is created, so
    :meth:`process` stores content rather than building a local index.

    Args:
        toolkit (Any): A configured
            :class:`~camel_goodmem.GoodMemToolkit`, which carries the
            connection, the spaces and any metadata filter.
    """

    def __init__(self, toolkit: Any) -> None:
        self.toolkit = toolkit

    def process(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        r"""Stores content in GoodMem so later queries can retrieve it.

        Args:
            content (str): The text to store.
            metadata (Optional[Dict[str, Any]]): Labels to attach to the
                memory. (default: :obj:`None`)
            **kwargs (Any): Ignored; accepted for interface compatibility.

        Returns:
            Dict[str, Any]: The created memory's ``memoryId`` and ``spaceId``.
        """
        return self.toolkit.goodmem_remember(content, metadata=metadata)

    def query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_RESULTS,
        similarity_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        r"""Retrieves chunks relevant to a query.

        Scores follow CAMEL's convention -- higher is a better match. GoodMem
        vector scores are negative distances and are flipped to match;
        reranker scores are already higher-is-better and are passed through
        unchanged, because negating one would invert the ranking.

        Args:
            query (str): The natural-language query.
            top_k (int): How many results to return. (default: :obj:`5`)
            similarity_threshold (Optional[float]): Drop results scoring
                below this value. There is no default: GoodMem's two score
                kinds are on different scales, and a reranker's scale is
                provider-dependent. (default: :obj:`None`)
            **kwargs (Any): Ignored; accepted for interface compatibility.

        Returns:
            List[Dict[str, Any]]: One dictionary per chunk, each with
                ``similarity score``, ``content path``, ``metadata``,
                ``extra_info`` and ``text``. When the retrieval was degraded
                and nothing usable came back, a single dictionary is returned
                whose ``text`` states what the server reported.
        """
        outcome = self.toolkit._retrieve(query, top_k)

        hits = outcome.hits
        if similarity_threshold is not None:
            kept = [
                h
                for h in hits
                if h.score is not None and h.score >= similarity_threshold
            ]
            if hits and not kept:
                observed = [h.score for h in hits if h.score is not None]
                warnings.warn(
                    f"similarity_threshold={similarity_threshold} removed "
                    f"all {len(hits)} result(s); observed scores ranged "
                    f"{min(observed):.4f}..{max(observed):.4f}. GoodMem "
                    "vector and reranker scores are not on a common 0-1 "
                    "scale.",
                    UserWarning,
                    stacklevel=2,
                )
            hits = kept

        results: list[dict[str, Any]] = []
        for hit in hits:
            extra: dict[str, Any] = {
                "goodmem_chunk_id": hit.chunk_id,
                "goodmem_memory_id": hit.memory_id,
                "goodmem_space_id": hit.space_id,
                "goodmem_score_kind": hit.score_kind,
                "goodmem_raw_score": hit.raw_score,
                "goodmem_partial": outcome.partial,
            }
            if outcome.partial:
                extra["goodmem_statuses"] = outcome.status_dicts
            results.append(
                {
                    "similarity score": (
                        str(hit.score) if hit.score is not None else ""
                    ),
                    "content path": hit.memory_id,
                    "metadata": hit.metadata,
                    "extra_info": extra,
                    "text": hit.text,
                }
            )

        if not results:
            # A bare list has no slot for a flag, so an empty degraded result
            # carries the server's reason in the one row it returns, and the
            # toolkit has already logged it at WARNING.
            if outcome.partial:
                warnings.warn(
                    outcome.warning_text(), UserWarning, stacklevel=2
                )
                return [
                    {
                        "text": (
                            "No results were returned and GoodMem reported a "
                            f"problem during retrieval: "
                            f"{outcome.warning_text()}"
                        ),
                        "extra_info": {
                            "goodmem_partial": True,
                            "goodmem_statuses": outcome.status_dicts,
                        },
                    }
                ]
            return [
                {
                    "text": (
                        f"No information relevant to {query!r} is stored in "
                        "the configured GoodMem space(s)."
                    ),
                    "extra_info": {"goodmem_partial": False},
                }
            ]
        return results
