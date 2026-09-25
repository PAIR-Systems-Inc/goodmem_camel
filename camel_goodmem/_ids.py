r"""Refuses any GoodMem id that is not a canonical UUID.

The ``goodmem`` SDK builds request paths by interpolating ids raw
(``f"/v1/memories/{id}"``), and ``httpx`` resolves dot segments before it
sends. So ``delete_memory("../spaces/<id>")`` became
``DELETE /v1/spaces/<id>``: it deleted a whole space and reported
``success: True``. Percent-encoding is no defence -- the server decodes
``%2e%2e`` back into a traversal.

Every GoodMem id -- memory, space, embedder, reranker -- is a UUID, so this
module accepts exactly that shape and nothing else, and every id is checked
here before any request is made. This is the only validator in the package.
"""

import re
import uuid
from typing import Annotated, Any

from pydantic import Field

#: A canonical, hyphenated UUID. Explicit ``[0-9a-fA-F]`` classes rather than
#: ``\d`` or ``\w``, which match non-ASCII digits.
UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{12}$"
)

# fullmatch, not match: with ``re.match`` a ``$`` also matches before a
# trailing newline, so "<uuid>\n" would pass.
_UUID_RE = re.compile(UUID_PATTERN)

#: A string parameter a model sees declared as a UUID. CAMEL builds each
#: tool's JSON schema from the signature, so the pattern reaches the model;
#: :func:`require_uuid` is what actually enforces it.
UuidStr = Annotated[str, Field(pattern=UUID_PATTERN)]


class GoodMemIdError(ValueError):
    r"""Raised when an id is not a UUID. No request has been made."""


def require_uuid(value: Any, field: str) -> str:
    r"""Returns ``value`` as a lower-case canonical UUID, or refuses it.

    Args:
        value (Any): The id as supplied -- by a model, a developer or
            configuration. A :class:`uuid.UUID` is accepted as well as a
            string.
        field (str): The argument's name, used in the error message.

    Returns:
        str: The id in lower-case canonical form, safe to put in a URL path.

    Raises:
        GoodMemIdError: If ``value`` is anything other than a hyphenated
            UUID: surrounding whitespace, braces, a ``urn:uuid:`` prefix, a
            path separator, a dot segment, a query or fragment, percent
            encoding, and the empty string are all refused.
    """
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, str) and _UUID_RE.fullmatch(value):
        return value.lower()
    shown = repr(value) if len(repr(value)) <= 80 else repr(value)[:77] + "..."
    raise GoodMemIdError(
        f"{field} must be a UUID such as "
        f"'01a0d44b-748d-72eb-b54e-c3ea2d956927'; got {shown}. GoodMem ids "
        "are UUIDs, and anything else could redirect the request to another "
        "resource, so it was refused before any request was made."
    )
