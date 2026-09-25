import os
import warnings
from pathlib import Path
from typing import Any

from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer, dependencies_required

from camel_goodmem._filters import all_of, resolve_filter
from camel_goodmem._results import (
    RetrievalOutcome,
    log_if_degraded,
    outcome_from_events,
)

from ._ids import UuidStr, require_uuid
from ._uploads import resolve_upload_path

logger = get_logger(__name__)

#: Bound on how many items a single listing call will pull, so a listing
#: cannot walk an entire server.
DEFAULT_MAX_LIST_ITEMS = 200

#: 0.2.0 read ``reranker_id=""`` as "no reranker", so a caller writing
#: ``reranker_id=os.getenv("X", "")`` meets this refusal at startup.
_NO_RERANKER_HINT = (
    "To search without a reranker, pass reranker_id=None or leave it out; "
    "an empty string is refused rather than read as 'no reranker'."
)


class GoodMemError(RuntimeError):
    r"""Raised when a GoodMem operation fails.

    Attributes:
        status_code (Optional[int]): The HTTP status the server returned,
            when the failure came from the server.
        body (Optional[str]): The server's response body, verbatim.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _wrap_api_error(exc: Exception, what: str) -> GoodMemError:
    r"""Converts an SDK error into a :class:`GoodMemError`, keeping the body.

    Args:
        exc (Exception): The exception the SDK raised.
        what (str): A short description of the operation that failed.

    Returns:
        GoodMemError: The wrapped error, carrying the server's own message.
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = str(exc)
    if body and body not in detail:
        detail = f"{detail} -- {body}"
    return GoodMemError(
        f"{what} failed: {detail}", status_code=status, body=body
    )


@MCPServer()
class GoodMemToolkit(BaseToolkit):
    r"""A toolkit for storing and recalling agent memory in GoodMem.

    GoodMem is a memory service for AI agents: documents are chunked,
    embedded and searched server-side. This toolkit wraps the official
    ``goodmem`` Python SDK and exposes a deliberately narrow set of tools to
    the model -- a search and, optionally, a write -- while every operational
    setting (which spaces, which reranker, whether uploads are possible) is
    fixed by the developer at construction time.

    Args:
        base_url (Optional[str]): The base URL of the GoodMem server. Falls
            back to the ``GOODMEM_BASE_URL`` environment variable.
            (default: :obj:`None`)
        api_key (Optional[str]): The GoodMem API key. Falls back to the
            ``GOODMEM_API_KEY`` environment variable. (default: :obj:`None`)
        space_ids (Optional[List[str]]): The UUIDs of the spaces this
            toolkit reads from and writes to. The model never chooses a
            space. (default: :obj:`None`)
        verify_ssl (bool): Whether to verify TLS certificates. Set to
            ``False`` only for a server with a self-signed certificate.
            (default: :obj:`True`)
        timeout (Optional[float]): Per-request timeout in seconds, applied to
            every HTTP call. (default: :obj:`30.0`)
        upload_dir (Optional[Union[str, Path]]): A directory that file
            uploads are confined to. When ``None``, no upload tool is offered
            and no path is ever read from disk. (default: :obj:`None`)
        reranker_id (Optional[str]): The UUID of a reranker to apply to
            retrieval. Without one, no relevance threshold is applied. Pass
            ``None`` for no reranker: an empty string is a malformed id and
            is refused. (default: :obj:`None`)
        min_score (Optional[float]): Drop hits scoring below this value.
            Applies only when ``reranker_id`` is set, because reranker scales
            are provider-dependent. Off by default. (default: :obj:`None`)
        metadata_filter (Optional[Union[Dict[str, Any], str]]): A filter
            every retrieved memory must match, applied server-side. Either a
            mapping, which must match as an ``AND`` of equalities, or an
            expression built with :mod:`camel_goodmem.filters` (``compare``,
            ``one_of``, ``not_equals``, ``any_of`` ...), sent verbatim. Set by
            the developer; the model never supplies a filter.
            (default: :obj:`None`)
        allow_write (bool): Whether the model may store new memories.
            (default: :obj:`True`)
        allow_admin_tools (bool): Whether space and embedder management is
            exposed to the model. (default: :obj:`False`)
        allow_delete (bool): Whether the model may delete memories.
            (default: :obj:`False`)
        max_list_items (int): Upper bound on items returned by a listing.
            (default: :obj:`200`)
        client (Optional[Any]): An already-configured ``goodmem.Goodmem``
            client. When supplied, its server, credentials and TLS settings
            are used as-is and it is never closed by this toolkit.
            (default: :obj:`None`)

    Every id -- configured here or passed to a method -- must be a UUID. The
    SDK puts ids into request paths unescaped, so anything else is refused
    with :class:`~camel_goodmem.GoodMemIdError` before a request is made.
    """

    @dependencies_required("goodmem")
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        space_ids: list[str] | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float | None = 30.0,
        upload_dir: str | Path | None = None,
        reranker_id: str | None = None,
        min_score: float | None = None,
        metadata_filter: dict[str, Any] | str | None = None,
        allow_write: bool = True,
        allow_admin_tools: bool = False,
        allow_delete: bool = False,
        max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
        client: Any | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        from goodmem import Goodmem

        self.base_url = (
            base_url or os.environ.get("GOODMEM_BASE_URL", "")
        ).rstrip("/")
        # The key is held privately: it is never an attribute a config dump,
        # a repr or a traceback can pick up.
        resolved_key = api_key or os.environ.get("GOODMEM_API_KEY", "")
        self.__api_key = resolved_key
        # Checked here so a misconfiguration fails at construction, and again
        # wherever they are used, because both are public attributes.
        self.space_ids = [
            require_uuid(space_id, f"space_ids[{i}]")
            for i, space_id in enumerate(space_ids or [])
        ]
        self.verify_ssl = verify_ssl
        self.reranker_id = (
            require_uuid(reranker_id, "reranker_id", hint=_NO_RERANKER_HINT)
            if reranker_id is not None
            else None
        )
        self.min_score = min_score
        # Resolved now so a bad filter fails at construction rather than on
        # the first search; resolved again at use, as it is public.
        resolve_filter(metadata_filter)
        self.metadata_filter: dict[str, Any] | str = (
            dict(metadata_filter)
            if isinstance(metadata_filter, dict)
            else metadata_filter or {}
        )
        self.allow_write = allow_write
        self.allow_admin_tools = allow_admin_tools
        self.allow_delete = allow_delete
        self.max_list_items = max_list_items
        self.upload_dir = (
            Path(upload_dir).expanduser().resolve()
            if upload_dir is not None
            else None
        )

        if client is not None:
            # An injected client carries its own server, credentials and TLS
            # settings. Demanding GOODMEM_BASE_URL/GOODMEM_API_KEY as well
            # would be asking for configuration this toolkit must not apply.
            self._client = client
            self._owns_client = False
        else:
            missing = [
                name
                for name, value in (
                    ("GOODMEM_API_KEY", resolved_key),
                    ("GOODMEM_BASE_URL", self.base_url),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Missing GoodMem credentials: {', '.join(missing)}. "
                    "Set them in the environment or pass api_key/base_url; "
                    "see https://docs.goodmem.ai. Alternatively pass an "
                    "already-configured client=Goodmem(...)."
                )
            self._client = Goodmem(
                base_url=self.base_url,
                api_key=resolved_key,
                timeout=timeout,
                verify=verify_ssl,
            )
            self._owns_client = True

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        r"""Returns a representation that never carries the API key."""
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"space_ids={self.space_ids!r})"
        )

    def close(self) -> None:
        r"""Closes the HTTP client, if this toolkit created it.

        A client passed in by the caller is left open: it belongs to whoever
        constructed it and may be shared with other components.
        """
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "GoodMemToolkit":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_spaces(self) -> list[str]:
        r"""Returns the configured spaces as canonical UUIDs.

        Raises:
            GoodMemError: If no space is configured.
            GoodMemIdError: If a configured space id is not a UUID.
        """
        if not self.space_ids:
            raise GoodMemError(
                "No space is configured. Construct the toolkit with "
                "space_ids=[...] so reads and writes have a destination."
            )
        return [
            require_uuid(space_id, f"space_ids[{i}]")
            for i, space_id in enumerate(self.space_ids)
        ]

    def _require_reranker(self) -> str | None:
        r"""Returns the configured reranker as a canonical UUID, if any."""
        if self.reranker_id is None:
            return None
        return require_uuid(
            self.reranker_id, "reranker_id", hint=_NO_RERANKER_HINT
        )

    def _space_keys(self, narrow: str = "") -> list[dict[str, Any]]:
        r"""Builds the ``spaceKeys`` payload, including any metadata filter.

        Args:
            narrow (str): A further expression that must also match, combined
                with the toolkit's own filter by ``AND``. (default: ``""``)
        """
        expression = all_of(resolve_filter(self.metadata_filter), narrow)
        keys: list[dict[str, Any]] = []
        for space_id in self._require_spaces():
            key: dict[str, Any] = {"spaceId": space_id}
            if expression:
                key["filter"] = expression
            keys.append(key)
        return keys

    def _retrieve(
        self, query: str, top_k: int, *, narrow: str = ""
    ) -> RetrievalOutcome:
        r"""Runs one retrieval and folds the stream into an outcome.

        Args:
            query (str): The natural-language query.
            top_k (int): How many chunks to ask the server for.
            narrow (str): A further filter expression, ANDed with the
                toolkit's own. (default: ``""``)

        Returns:
            RetrievalOutcome: The hits and any statuses the server reported.
        """
        reranker_id = self._require_reranker()
        kwargs: dict[str, Any] = {
            "message": query,
            "space_keys": self._space_keys(narrow),
            "requested_size": top_k,
            "fetch_memory": True,
        }
        if reranker_id:
            kwargs["reranker_id"] = reranker_id

        try:
            stream = self._client.memories.retrieve(**kwargs)
            with stream as events:
                outcome = outcome_from_events(
                    events, reranked=bool(reranker_id)
                )
        except GoodMemError:
            raise
        except Exception as exc:
            raise _wrap_api_error(exc, "Retrieval") from exc

        if self.min_score is not None and reranker_id:
            kept = [
                h
                for h in outcome.hits
                if h.score is not None and h.score >= self.min_score
            ]
            if outcome.hits and not kept:
                observed = [
                    h.score for h in outcome.hits if h.score is not None
                ]
                warnings.warn(
                    f"min_score={self.min_score} removed all "
                    f"{len(outcome.hits)} reranked result(s); observed scores "
                    f"ranged {min(observed):.4f}..{max(observed):.4f}. "
                    "Reranker scales are provider-dependent, not 0-1.",
                    UserWarning,
                    stacklevel=3,
                )
            outcome.hits = kept

        log_if_degraded(outcome, "goodmem_search")
        return outcome

    # ------------------------------------------------------------------
    # model-facing: search
    # ------------------------------------------------------------------

    def goodmem_search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        r"""Searches stored memories for information relevant to a question.

        Use this to recall facts, documents or past context that were saved
        earlier. Results are ranked by relevance, best first.

        Args:
            query (str): A natural-language description of what to find.
            top_k (int): How many results to return. (default: :obj:`5`)

        Returns:
            Dict[str, Any]: A dictionary with ``results`` (a list of matching
                chunks, each with its text and the metadata of the memory it
                came from), ``partial`` (``True`` when the server reported a
                problem during this search), ``statuses`` (what the server
                reported), and ``query``.
        """
        outcome = self._retrieve(query, top_k)
        result: dict[str, Any] = {
            "success": True,
            "query": query,
            "results": [h.as_dict() for h in outcome.hits],
            "totalResults": len(outcome.hits),
            "partial": outcome.partial,
            "statuses": outcome.status_dicts,
            "resultSetId": outcome.result_set_id,
        }
        if outcome.partial:
            # Q4a and Q4b: hits are never discarded because of a status, and
            # an empty result carries the reason rather than reading as a
            # clean miss.
            result["warning"] = outcome.warning_text()
        if outcome.abstract_reply:
            result["abstractReply"] = outcome.abstract_reply
        return result

    # ------------------------------------------------------------------
    # model-facing: write
    # ------------------------------------------------------------------

    def goodmem_remember(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r"""Stores a piece of text as a memory for later recall.

        Args:
            text (str): The text to remember.
            metadata (Optional[Dict[str, Any]]): Optional key-value labels to
                attach, so the memory can be filtered later.
                (default: :obj:`None`)

        Returns:
            Dict[str, Any]: A dictionary with ``success``, ``memoryId`` and
                ``spaceId``.
        """
        space_id = self._require_spaces()[0]
        try:
            memory = self._client.memories.create(
                space_id=space_id,
                original_content=text,
                content_type="text/plain",
                metadata=metadata or None,
            )
        except Exception as exc:
            raise _wrap_api_error(exc, "Creating a memory") from exc
        return {
            "success": True,
            "memoryId": str(getattr(memory, "memory_id", "") or ""),
            "spaceId": str(getattr(memory, "space_id", "") or space_id),
            "status": str(getattr(memory, "processing_status", "") or ""),
        }

    def goodmem_upload_file(
        self, file_name: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        r"""Stores a file from the configured upload directory as a memory.

        Only files inside the directory the developer configured can be
        uploaded; any other path is refused.

        Args:
            file_name (str): The name of a file inside the upload directory.
            metadata (Optional[Dict[str, Any]]): Optional key-value labels to
                attach to the memory. (default: :obj:`None`)

        Returns:
            Dict[str, Any]: A dictionary with ``success``, ``memoryId``,
                ``spaceId`` and the ``fileName`` that was stored.
        """
        space_id = self._require_spaces()[0]
        resolved = resolve_upload_path(file_name, self.upload_dir)
        try:
            memory = self._client.memories.create(
                space_id=space_id,
                file_path=str(resolved),
                metadata=metadata or None,
            )
        except Exception as exc:
            raise _wrap_api_error(exc, "Uploading a file") from exc
        return {
            "success": True,
            "memoryId": str(getattr(memory, "memory_id", "") or ""),
            "spaceId": str(getattr(memory, "space_id", "") or space_id),
            "fileName": resolved.name,
        }

    # ------------------------------------------------------------------
    # developer / admin surface
    # ------------------------------------------------------------------

    def get_memory(
        self, memory_id: UuidStr, include_content: bool = False
    ) -> dict[str, Any]:
        r"""Fetches one memory by id, optionally with its original content.

        The content is decoded according to the memory's own content type:
        text is returned as text, anything else as base64. A content fetch
        that fails is an error, not a successful result with a note in it.

        Args:
            memory_id (str): The UUID of the memory.
            include_content (bool): Whether to fetch the original content as
                well. (default: :obj:`False`)

        Returns:
            Dict[str, Any]: A dictionary with ``success``, ``memory`` and,
                when requested, ``content`` plus ``contentEncoding``.
        """
        memory_id = require_uuid(memory_id, "memory_id")
        try:
            memory = self._client.memories.get(id=memory_id)
        except Exception as exc:
            raise _wrap_api_error(exc, f"Fetching memory {memory_id}") from exc

        dump = getattr(memory, "model_dump", None)
        payload = dump(by_alias=True, exclude_none=True) if dump else {}
        result: dict[str, Any] = {"success": True, "memory": payload}

        if include_content:
            try:
                raw = self._client.memories.content(id=memory_id)
            except Exception as exc:
                raise _wrap_api_error(
                    exc, f"Fetching content of memory {memory_id}"
                ) from exc
            content_type = str(
                payload.get("contentType") or payload.get("content_type") or ""
            )
            result["content"], result["contentEncoding"] = _decode_content(
                raw, content_type
            )
        return result

    def list_spaces(self) -> list[dict[str, Any]]:
        r"""Lists spaces, following pagination up to ``max_list_items``.

        Returns:
            List[Dict[str, Any]]: Space objects with ``spaceId`` and ``name``.
        """
        try:
            page = self._client.spaces.list(max_items=self.max_list_items)
            spaces = list(page)
        except Exception as exc:
            raise _wrap_api_error(exc, "Listing spaces") from exc
        return [
            {
                "spaceId": str(getattr(s, "space_id", "") or ""),
                "name": str(getattr(s, "name", "") or ""),
            }
            for s in spaces
        ]

    def list_embedders(self) -> list[dict[str, Any]]:
        r"""Lists the embedder models available on the server.

        Returns:
            List[Dict[str, Any]]: Embedder objects with ``embedderId``,
                ``displayName`` and ``modelIdentifier``.
        """
        try:
            page = self._client.embedders.list()
            embedders = list(page)
        except Exception as exc:
            raise _wrap_api_error(exc, "Listing embedders") from exc
        return [
            {
                "embedderId": str(getattr(e, "embedder_id", "") or ""),
                "displayName": str(getattr(e, "display_name", "") or ""),
                "modelIdentifier": str(
                    getattr(e, "model_identifier", "") or ""
                ),
            }
            for e in embedders
        ]

    def create_space(self, name: str, embedder_id: UuidStr) -> dict[str, Any]:
        r"""Creates a space, or reuses one whose embedder already matches.

        A space cannot change embedder after creation, so reusing a space by
        name alone silently writes vectors from a different model than the
        caller asked for. Reuse therefore requires the embedder to match, and
        a mismatch is an error naming both.

        Args:
            name (str): The space name.
            embedder_id (str): The UUID of the embedder the space must use.

        Returns:
            Dict[str, Any]: A dictionary with ``success``, ``spaceId``,
                ``name``, ``embedderId`` and ``reused``.

        Raises:
            GoodMemError: If a space of that name exists with a different
                embedder, or if several spaces share the name.
        """
        embedder_id = require_uuid(embedder_id, "embedder_id")
        try:
            existing = [
                s
                for s in self._client.spaces.list(
                    max_items=self.max_list_items
                )
                if str(getattr(s, "name", "") or "") == name
            ]
        except Exception as exc:
            raise _wrap_api_error(exc, "Listing spaces") from exc

        if len(existing) > 1:
            raise GoodMemError(
                f"{len(existing)} spaces are named {name!r}; refusing to "
                "guess which one was meant. Pass a space id instead."
            )
        if existing:
            space = existing[0]
            actual = _space_embedder_ids(space)
            if embedder_id not in actual:
                raise GoodMemError(
                    f"Space {name!r} already exists and is indexed by "
                    f"embedder(s) {actual}, not {embedder_id!r}. An embedder "
                    "cannot be changed after creation; use a different name "
                    "or the embedder the space was built with."
                )
            return {
                "success": True,
                "spaceId": str(getattr(space, "space_id", "") or ""),
                "name": name,
                "embedderId": embedder_id,
                "reused": True,
            }

        try:
            space = self._client.spaces.create(
                name=name,
                space_embedders=[
                    {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
                ],
            )
        except Exception as exc:
            raise _wrap_api_error(exc, f"Creating space {name!r}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or ""),
            "name": str(getattr(space, "name", "") or name),
            "embedderId": embedder_id,
            "reused": False,
        }

    def goodmem_get_space(self, space_id: UuidStr) -> dict[str, Any]:
        r"""Fetches one space by id.

        Args:
            space_id (str): The UUID of the space.

        Returns:
            dict[str, Any]: The space, with ``spaceId``, ``name``,
                ``embedderIds`` and ``labels``.
        """
        space_id = require_uuid(space_id, "space_id")
        try:
            space = self._client.spaces.get(id=space_id)
        except Exception as exc:
            raise _wrap_api_error(exc, f"Fetching space {space_id}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or ""),
            "name": str(getattr(space, "name", "") or ""),
            "embedderIds": _space_embedder_ids(space),
            "labels": dict(getattr(space, "labels", None) or {}),
        }

    def update_space(
        self,
        space_id: UuidStr,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        replace_labels: bool = False,
    ) -> dict[str, Any]:
        r"""Renames a space or edits its labels.

        ``publicRead`` is deliberately not offered: the server removed the
        field and rejects any request carrying it with HTTP 400.

        Args:
            space_id (str): The UUID of the space to update.
            name (str | None): A new name. (default: :obj:`None`)
            labels (dict[str, str] | None): Labels to merge, or to replace
                with when ``replace_labels`` is set. (default: :obj:`None`)
            replace_labels (bool): Replace all labels instead of merging.
                (default: :obj:`False`)

        Returns:
            dict[str, Any]: A dictionary with ``success``, ``spaceId`` and
                ``name``.
        """
        space_id = require_uuid(space_id, "space_id")
        request: dict[str, Any] = {}
        if name is not None:
            request["name"] = name
        if labels is not None:
            key = "replaceLabels" if replace_labels else "mergeLabels"
            request[key] = dict(labels)
        if not request:
            raise GoodMemError(
                "update_space() needs a name or labels to change."
            )
        try:
            space = self._client.spaces.update(id=space_id, request=request)
        except Exception as exc:
            raise _wrap_api_error(exc, f"Updating space {space_id}") from exc
        return {
            "success": True,
            "spaceId": str(getattr(space, "space_id", "") or space_id),
            "name": str(getattr(space, "name", "") or ""),
        }

    def delete_space(self, space_id: UuidStr) -> dict[str, Any]:
        r"""Permanently deletes a space and every memory in it.

        Args:
            space_id (str): The UUID of the space to delete.

        Returns:
            dict[str, Any]: A dictionary with ``success`` and ``spaceId``.
        """
        space_id = require_uuid(space_id, "space_id")
        try:
            self._client.spaces.delete(id=space_id)
        except Exception as exc:
            raise _wrap_api_error(exc, f"Deleting space {space_id}") from exc
        return {"success": True, "spaceId": space_id}

    def list_memories(
        self, space_id: UuidStr | None = None
    ) -> list[dict[str, Any]]:
        r"""Lists memories in a space, following pagination.

        Args:
            space_id (str | None): The UUID of the space to list. Defaults to
                the first configured space. (default: :obj:`None`)

        Returns:
            list[dict[str, Any]]: Memory records with ``memoryId``,
                ``spaceId``, ``contentType``, ``processingStatus`` and
                ``metadata``.
        """
        # An empty string is a malformed id, not a request for the default.
        target = (
            self._require_spaces()[0]
            if space_id is None
            else require_uuid(space_id, "space_id")
        )
        try:
            page = self._client.memories.list(
                space_id=target, max_items=self.max_list_items
            )
            memories = list(page)
        except Exception as exc:
            raise _wrap_api_error(exc, "Listing memories") from exc
        return [
            {
                "memoryId": str(getattr(m, "memory_id", "") or ""),
                "spaceId": str(getattr(m, "space_id", "") or ""),
                "contentType": str(getattr(m, "content_type", "") or ""),
                "processingStatus": str(
                    getattr(m, "processing_status", "") or ""
                ),
                "metadata": dict(getattr(m, "metadata", None) or {}),
            }
            for m in memories
        ]

    def delete_memory(self, memory_id: UuidStr) -> dict[str, Any]:
        r"""Permanently deletes a memory and everything derived from it.

        Args:
            memory_id (str): The UUID of the memory to delete.

        Returns:
            Dict[str, Any]: A dictionary with ``success`` and ``memoryId``.
        """
        memory_id = require_uuid(memory_id, "memory_id")
        try:
            self._client.memories.delete(id=memory_id)
        except Exception as exc:
            raise _wrap_api_error(exc, f"Deleting memory {memory_id}") from exc
        return {"success": True, "memoryId": memory_id}

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------

    def get_tools(self) -> list[FunctionTool]:
        r"""Returns the tools this toolkit exposes to a model.

        The default surface is a search, plus a write when ``allow_write`` is
        set. Space and embedder management, deletion and file upload are each
        opt-in, because a model does not need to administer a memory server
        in order to use one.

        Returns:
            List[FunctionTool]: The tools a model may call.
        """
        tools: list[FunctionTool] = [FunctionTool(self.goodmem_search)]
        if self.allow_write:
            tools.append(FunctionTool(self.goodmem_remember))
        if self.upload_dir is not None:
            tools.append(FunctionTool(self.goodmem_upload_file))
        if self.allow_admin_tools:
            tools.extend(
                [
                    FunctionTool(self.list_spaces),
                    FunctionTool(self.list_embedders),
                    FunctionTool(self.goodmem_get_space),
                    FunctionTool(self.create_space),
                    FunctionTool(self.update_space),
                    FunctionTool(self.list_memories),
                    FunctionTool(self.get_memory),
                ]
            )
        if self.allow_delete:
            tools.extend(
                [
                    FunctionTool(self.delete_memory),
                    FunctionTool(self.delete_space),
                ]
            )
        return tools


def _space_embedder_ids(space: Any) -> list[str]:
    r"""Returns the embedder ids a space is actually indexed by.

    Args:
        space (Any): A space model from the SDK.

    Returns:
        List[str]: The embedder ids, as strings.
    """
    configs = getattr(space, "space_embedders", None) or []
    ids: list[str] = []
    for config in configs:
        value = getattr(config, "embedder_id", None)
        if value is None and isinstance(config, dict):
            value = config.get("embedderId") or config.get("embedder_id")
        if value:
            ids.append(str(value))
    return ids


def _decode_content(raw: bytes, content_type: str) -> "tuple[Any, str]":
    r"""Decodes memory content according to its content type.

    Args:
        raw (bytes): The bytes the server returned.
        content_type (str): The memory's content type, with any charset.

    Returns:
        tuple[Any, str]: The decoded content and the encoding used, either
            ``"text"`` or ``"base64"``.
    """
    import base64

    primary = (content_type or "").split(";")[0].strip().lower()
    charset = "utf-8"
    for part in (content_type or "").split(";")[1:]:
        if "charset=" in part:
            charset = part.split("charset=", 1)[1].strip() or "utf-8"

    textual = primary.startswith("text/") or primary in {
        "application/json",
        "application/xml",
        "application/javascript",
    }
    if textual:
        try:
            return raw.decode(charset), "text"
        except (UnicodeDecodeError, LookupError):
            # Declared as text but not decodable: keep the bytes rather than
            # substituting replacement characters into the content.
            return base64.b64encode(raw).decode("ascii"), "base64"
    return base64.b64encode(raw).decode("ascii"), "base64"
