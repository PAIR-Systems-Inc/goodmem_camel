r"""Shared handling of a GoodMem retrieval stream.

Both :class:`~camel_goodmem.GoodMemToolkit` and
:class:`~camel_goodmem.GoodMemRetriever` consume retrieval through this
module, so the two cannot drift apart in how they classify a status, join a
chunk to its memory, or orient a score.
"""

from dataclasses import dataclass, field
from typing import Any

from camel.logger import get_logger

logger = get_logger(__name__)

#: Status codes that report an *optional* feature the caller never configured.
#: The server files both under "Informational status messages (non-error)":
#: nothing the caller asked for is missing, so they are noise unconditionally
#: and their ``details`` are never inspected to decide.
INFORMATIONAL_CODES = frozenset(
    {"LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED"}
)

#: The status the server sends when a requested reranker could not run. The
#: server then returns the vector-stage hits instead, scored as negative
#: distances rather than on the reranker's scale.
RERANKING_FAILED_CODE = "RERANKING_FAILED"

#: Surfaced in place of a code this build of the SDK does not recognise.
UNKNOWN_CODE = "UNKNOWN"

#: Reported when the stream ended mid-line or carried an undecodable line.
MALFORMED_STREAM_CODE = "MALFORMED_STREAM"


@dataclass
class RetrievalStatus:
    r"""One status event reported by the server during a retrieval.

    Attributes:
        code (str): The server's status code, or ``"UNKNOWN"`` when this
            build does not recognise the code the server sent.
        message (str): The server's own human-readable message.
        details (Dict[str, Any]): Any structured detail the server attached.
        informational (bool): Whether this status reports an unconfigured
            optional feature rather than a problem.
    """

    code: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    informational: bool = False

    def as_dict(self) -> dict[str, Any]:
        r"""Returns the status as a plain JSON-safe dictionary."""
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


@dataclass
class RetrievalHit:
    r"""One chunk, joined to the memory it came from.

    Attributes:
        chunk_id (str): The chunk's UUID.
        text (str): The chunk's text.
        memory_id (str): The UUID of the memory the chunk belongs to.
        score (Optional[float]): Relevance oriented so that higher is better.
        raw_score (Optional[float]): The score exactly as the server sent it.
        score_kind (str): ``"vector"`` or ``"reranker"`` -- the stage the
            score came from. The two are not on a common scale.
        space_id (str): The space the memory lives in.
        metadata (Dict[str, Any]): The memory's metadata.
        content_type (str): The memory's content type.
    """

    chunk_id: str
    text: str
    memory_id: str = ""
    score: float | None = None
    raw_score: float | None = None
    score_kind: str = "vector"
    space_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    content_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        r"""Returns the hit as a plain JSON-safe dictionary."""
        return {
            "chunkId": self.chunk_id,
            "text": self.text,
            "memoryId": self.memory_id,
            "spaceId": self.space_id,
            "score": self.score,
            "rawScore": self.raw_score,
            "scoreKind": self.score_kind,
            "contentType": self.content_type,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalOutcome:
    r"""Everything one retrieval produced.

    Attributes:
        hits (List[RetrievalHit]): The chunks that came back, in server order.
        statuses (List[RetrievalStatus]): Every non-informational status the
            server reported.
        partial (bool): ``True`` when the server reported a real problem
            during this retrieval. Independent of whether hits came back.
        result_set_id (str): The server's identifier for this result set.
        abstract_reply (Optional[str]): The LLM summary, when one was asked
            for and produced.
        reranked (bool): Whether the hits carry reranker scores. Decided from
            what the server reported, not from configuration: when a reranker
            was requested but failed, the hits are the vector fallback.
    """

    hits: list[RetrievalHit] = field(default_factory=list)
    statuses: list[RetrievalStatus] = field(default_factory=list)
    partial: bool = False
    result_set_id: str = ""
    abstract_reply: str | None = None
    reranked: bool = False

    @property
    def status_dicts(self) -> list[dict[str, Any]]:
        r"""Returns the statuses as plain dictionaries."""
        return [s.as_dict() for s in self.statuses]

    def warning_text(self) -> str:
        r"""Returns a one-line summary of why this retrieval was degraded."""
        if not self.statuses:
            return ""
        parts = [
            f"{s.code}: {s.message}" if s.message else s.code
            for s in self.statuses
        ]
        return "GoodMem reported a problem during retrieval -- " + "; ".join(
            parts
        )


def classify_status(raw_code: str | None, message: str) -> RetrievalStatus:
    r"""Classifies one status event from a retrieval stream.

    Implements Q1 and Q3 of the retrieval status contract: the two
    informational codes are noise unconditionally, and a code this build does
    not recognise is surfaced as ``UNKNOWN`` rather than dropped or raised on.

    Args:
        raw_code (Optional[str]): The status code as decoded from the stream.
            ``None`` means the SDK did not recognise the code the server sent.
        message (str): The server's message for the status.

    Returns:
        RetrievalStatus: The classified status.
    """
    if raw_code is None:
        # Q3: a server upgrade must not silently change behaviour. The typed
        # model discards the unrecognised string, so the code is reported as
        # UNKNOWN and the server's message is kept as the only description.
        return RetrievalStatus(
            code=UNKNOWN_CODE, message=message, informational=False
        )
    if raw_code in INFORMATIONAL_CODES:
        return RetrievalStatus(
            code=raw_code, message=message, informational=True
        )
    return RetrievalStatus(code=raw_code, message=message)


def reranking_failed(statuses: "list[RetrievalStatus]") -> bool:
    r"""Returns whether the server reported that reranking did not happen.

    ``RERANKING_FAILED`` says so directly. A ``NOT_FOUND`` naming the
    reranker (live: ``details: {"reranker_id": ...}``, message "Reranker not
    found") means the same, even if it arrives alone.

    Args:
        statuses (List[RetrievalStatus]): The statuses of one retrieval.

    Returns:
        bool: ``True`` if the hits cannot be reranker-scored.
    """
    for status in statuses:
        if status.code == RERANKING_FAILED_CODE:
            return True
        if status.code == "NOT_FOUND" and (
            "reranker_id" in status.details
            or "rerankerId" in status.details
            or "reranker" in status.message.lower()
        ):
            return True
    return False


def _as_dict(value: Any) -> dict[str, Any]:
    r"""Best-effort conversion of an SDK model or mapping to a dictionary."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(by_alias=True, exclude_none=True)
        except TypeError:
            return dump()
    return {}


def _getattr_any(obj: Any, *names: str) -> Any:
    r"""Returns the first present attribute or mapping key from ``names``."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            got = getattr(obj, name, None)
            if got is not None:
                return got
    return None


def outcome_from_events(
    events: Any,
    *,
    reranked: bool = False,
) -> RetrievalOutcome:
    r"""Folds a GoodMem retrieval stream into a :class:`RetrievalOutcome`.

    Chunks are joined to their memory by memory UUID rather than by the
    positional ``memoryIndex`` the stream also carries, so a reordered or
    partial stream cannot attach a chunk to the wrong memory. Chunks are
    de-duplicated by chunk id -- never by memory id, which would collapse
    distinct chunks of one document.

    Args:
        events (Any): An iterable of retrieval events from the SDK.
        reranked (bool): Whether a reranker was requested. The hits are
            scored as reranked only if the server did not also report that
            reranking failed -- then they are the vector fallback, negative
            distances, and are oriented as such. (default: :obj:`False`)

    Returns:
        RetrievalOutcome: The hits, the statuses, and whether the retrieval
            was degraded.
    """
    outcome = RetrievalOutcome()
    memories: dict[str, dict[str, Any]] = {}
    pending: list[tuple[RetrievalHit, str]] = []
    seen_chunks: set = set()

    for event in _iter_guarded(events, outcome):
        boundary = _getattr_any(event, "result_set_boundary")
        if boundary is not None:
            rsid = _getattr_any(boundary, "result_set_id", "resultSetId")
            if rsid:
                outcome.result_set_id = str(rsid)
            continue

        status = _getattr_any(event, "status")
        if status is not None:
            parsed = classify_status(
                _getattr_any(status, "code"),
                str(_getattr_any(status, "message") or ""),
            )
            parsed.details = _as_dict(_getattr_any(status, "details"))
            if not parsed.informational:
                outcome.statuses.append(parsed)
                outcome.partial = True
            continue

        definition = _getattr_any(event, "memory_definition")
        if definition is not None:
            mem = _as_dict(definition)
            mem_id = str(mem.get("memoryId") or mem.get("memory_id") or "")
            if mem_id:
                memories[mem_id] = mem
            continue

        reply = _getattr_any(event, "abstract_reply")
        if reply is not None:
            text = _getattr_any(reply, "reply", "text", "content")
            outcome.abstract_reply = (
                str(text) if text is not None else str(reply)
            )
            continue

        item = _getattr_any(event, "retrieved_item")
        if item is None:
            continue
        chunk_ref = _getattr_any(item, "chunk")
        if chunk_ref is None:
            continue
        inner = _getattr_any(chunk_ref, "chunk")
        if inner is None:
            continue

        chunk_id = str(_getattr_any(inner, "chunk_id", "chunkId") or "")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        raw_score = _getattr_any(
            chunk_ref, "relevance_score", "relevanceScore"
        )
        raw_value = float(raw_score) if raw_score is not None else None
        memory_id = str(_getattr_any(inner, "memory_id", "memoryId") or "")
        hit = RetrievalHit(
            chunk_id=chunk_id,
            text=str(_getattr_any(inner, "chunk_text", "chunkText") or ""),
            memory_id=memory_id,
            raw_score=raw_value,
        )
        pending.append((hit, memory_id))

    # Decided once the whole stream is in: a RERANKING_FAILED can follow the
    # hits it applies to.
    outcome.reranked = reranked and not reranking_failed(outcome.statuses)
    for hit, memory_id in pending:
        hit.score = orient_score(hit.raw_score, reranked=outcome.reranked)
        hit.score_kind = "reranker" if outcome.reranked else "vector"
        mem = memories.get(memory_id) or {}
        if mem:
            hit.space_id = str(mem.get("spaceId") or mem.get("space_id") or "")
            hit.content_type = str(
                mem.get("contentType") or mem.get("content_type") or ""
            )
            meta = mem.get("metadata")
            hit.metadata = dict(meta) if isinstance(meta, dict) else {}
        outcome.hits.append(hit)

    return outcome


def _iter_guarded(events: Any, outcome: RetrievalOutcome) -> Any:
    r"""Iterates a retrieval stream, surviving a stream that ends badly.

    A truncated or undecodable stream is a real problem, but what already
    arrived is still valid: it is kept, and the failure is reported as a
    ``MALFORMED_STREAM`` status rather than raised, so a partial answer is
    never silently presented as a complete one.

    Args:
        events (Any): The event iterable from the SDK.
        outcome (RetrievalOutcome): The outcome being built, annotated in
            place if the stream fails.

    Yields:
        Any: Each event that decoded successfully.
    """
    iterator = iter(events)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception as exc:
            outcome.statuses.append(
                RetrievalStatus(
                    code=MALFORMED_STREAM_CODE,
                    message=(
                        "The retrieval stream ended badly; results may be "
                        f"incomplete: {exc}"
                    ),
                )
            )
            outcome.partial = True
            return


def orient_score(raw: float | None, *, reranked: bool) -> float | None:
    r"""Returns a score oriented so that a higher number is a better match.

    GoodMem's vector scores are negative distances -- ``-0.51`` is a closer
    match than ``-0.88`` -- while reranker scores are already higher-is-better
    on a provider-dependent scale. Negating a reranker score would invert the
    ranking, so only vector scores are flipped.

    Args:
        raw (Optional[float]): The score exactly as the server sent it.
        reranked (bool): Whether the score came from a reranker.

    Returns:
        Optional[float]: The oriented score, or ``None`` if there was none.
    """
    if raw is None:
        return None
    return raw if reranked else -raw


def log_if_degraded(outcome: RetrievalOutcome, where: str) -> None:
    r"""Emits a WARNING log line when a retrieval came back degraded.

    Args:
        outcome (RetrievalOutcome): The outcome to inspect.
        where (str): A short label for the call site, used in the log line.
    """
    if outcome.partial:
        logger.warning("%s: %s", where, outcome.warning_text())
