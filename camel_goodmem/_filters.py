r"""Builds GoodMem metadata filter expressions safely.

GoodMem filters are expressions evaluated server-side, not SQL. Interpolating
a caller's value into one is both a filter-injection hole and a correctness
bug: an ordinary apostrophe produces a malformed expression. This module is
the only place in the package that writes a filter string.

The escaping and casting rules below were verified live against GoodMem
server v1.0.320:

- a literal is single-quoted; ``'`` inside it escapes as ``\'`` and a literal
  backslash as ``\\``. SQL-style ``''`` doubling and double-quoted strings
  are both rejected with HTTP 400.
- a raw newline inside a literal is rejected, so control characters are
  refused here rather than sent.
- ``val()`` yields JSON, so a comparison must cast to the stored type:
  ``TEXT`` for strings, ``NUMERIC`` for numbers, ``BOOLEAN`` for booleans.
  **The cast has to match:** comparing a boolean as ``TEXT`` is accepted with
  HTTP 200 and matches nothing.
"""

import re
from collections.abc import Iterable
from typing import Any

#: Field names are restricted rather than escaped -- the JSONPath member
#: grammar is not worth trying to quote around.
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class GoodMemFilterError(ValueError):
    r"""Raised when a filter cannot be expressed safely."""


def escape_literal(value: str) -> str:
    r"""Quotes a string as a GoodMem filter literal.

    Args:
        value (str): The raw value to quote.

    Returns:
        str: The quoted literal, ready to embed in an expression.

    Raises:
        GoodMemFilterError: If the value contains a control character, which
            the server rejects inside a literal.
    """
    if _CONTROL_CHARS.search(value):
        raise GoodMemFilterError(
            "Filter values cannot contain control characters (including "
            "newlines and tabs); the server rejects them inside a literal."
        )
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _check_field(field: str) -> str:
    r"""Validates a metadata field name.

    Args:
        field (str): The metadata key to filter on.

    Returns:
        str: The validated field name.

    Raises:
        GoodMemFilterError: If the name is outside the safe character set.
    """
    if not _SAFE_FIELD.match(field or ""):
        raise GoodMemFilterError(
            f"Unsupported metadata field name {field!r}. Field names may "
            "contain letters, digits, underscore, dot and hyphen, and must "
            "start with a letter or underscore."
        )
    return field


def _accessor(field: str, cast: str) -> str:
    r"""Returns the ``CAST(val(...))`` accessor for a field."""
    return f"CAST(val('$.{_check_field(field)}') AS {cast})"


def _render(value: Any) -> "tuple[str, str]":
    r"""Returns the cast and rendered literal for a Python value.

    Booleans are checked before numbers because :obj:`bool` is a subclass of
    :obj:`int`; stringifying a boolean would produce a filter that the server
    accepts and that matches nothing.

    Args:
        value (Any): The value to compare against.

    Returns:
        tuple[str, str]: The SQL-ish cast name and the rendered literal.

    Raises:
        GoodMemFilterError: If the value has no safe representation.
    """
    if isinstance(value, bool):
        return "BOOLEAN", "true" if value else "false"
    if isinstance(value, (int, float)):
        return "NUMERIC", repr(value)
    if isinstance(value, str):
        return "TEXT", escape_literal(value)
    raise GoodMemFilterError(
        f"Unsupported filter value type {type(value).__name__}; use str, "
        "int, float or bool."
    )


def equals(field: str, value: Any) -> str:
    r"""Builds an equality filter for one metadata field.

    Args:
        field (str): The metadata key.
        value (Any): The value to match. ``str``, ``int``, ``float`` and
            ``bool`` are supported, each cast to the type GoodMem stores.

    Returns:
        str: A filter expression.
    """
    cast, literal = _render(value)
    return f"{_accessor(field, cast)} = {literal}"


def not_equals(field: str, value: Any) -> str:
    r"""Builds an inequality filter for one metadata field.

    Args:
        field (str): The metadata key.
        value (Any): The value that must not match.

    Returns:
        str: A filter expression.
    """
    cast, literal = _render(value)
    return f"{_accessor(field, cast)} != {literal}"


def compare(field: str, operator: str, value: Any) -> str:
    r"""Builds an ordering comparison against a numeric metadata field.

    Args:
        field (str): The metadata key.
        operator (str): One of ``>``, ``>=``, ``<`` or ``<=``.
        value (Any): A number to compare against.

    Returns:
        str: A filter expression.

    Raises:
        GoodMemFilterError: If the operator or the value type is unsupported.
    """
    if operator not in {">", ">=", "<", "<="}:
        raise GoodMemFilterError(
            f"Unsupported comparison operator {operator!r}."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoodMemFilterError("Ordering comparisons apply to numbers only.")
    return f"{_accessor(field, 'NUMERIC')} {operator} {value!r}"


def one_of(field: str, values: Iterable[Any]) -> str:
    r"""Builds an ``IN`` filter for one metadata field.

    Args:
        field (str): The metadata key.
        values (Iterable[Any]): The values to match. All must share one type.

    Returns:
        str: A filter expression.

    Raises:
        GoodMemFilterError: If the values are empty or of mixed types.
    """
    rendered: list[str] = []
    casts = set()
    for value in values:
        cast, literal = _render(value)
        casts.add(cast)
        rendered.append(literal)
    if not rendered:
        raise GoodMemFilterError("one_of() needs at least one value.")
    if len(casts) > 1:
        raise GoodMemFilterError(
            "one_of() values must all be of the same type."
        )
    accessor = _accessor(field, casts.pop())
    return f"{accessor} IN ({', '.join(rendered)})"


def all_of(*expressions: str | None) -> str:
    r"""Combines filter expressions with ``AND``.

    Args:
        expressions (Optional[str]): The expressions to combine. ``None`` and
            empty expressions are skipped.

    Returns:
        str: The combined expression, or ``""`` if nothing was supplied.
    """
    parts = [e for e in expressions if e]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({p})" for p in parts)


def any_of(*expressions: str | None) -> str:
    r"""Combines filter expressions with ``OR``.

    Args:
        expressions (Optional[str]): The expressions to combine. ``None`` and
            empty expressions are skipped.

    Returns:
        str: The combined expression, or ``""`` if nothing was supplied.
    """
    parts = [e for e in expressions if e]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " OR ".join(f"({p})" for p in parts)


def from_mapping(metadata: dict[str, Any]) -> str:
    r"""Builds an ``AND`` of equality filters from a mapping.

    This is the convenience form used by the toolkit and the retriever, so a
    developer can pass ``{"tenant": "acme", "year": 2026}`` without writing an
    expression by hand.

    Args:
        metadata (Dict[str, Any]): Field/value pairs that must all match.

    Returns:
        str: A filter expression, or ``""`` for an empty mapping.
    """
    if not metadata:
        return ""
    return all_of(*(equals(k, v) for k, v in sorted(metadata.items())))
