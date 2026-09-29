r"""Confines model-supplied upload paths to one directory.

A path that reaches an upload call is chosen by a language model. Without a
boundary, ``/etc/hostname`` or the caller's own credentials file is a valid
argument, and the file is read and uploaded. Uploads therefore exist only
when the developer configures ``upload_dir``, and every path is resolved --
symlinks included -- before it is compared against that directory.
"""

import os
from pathlib import Path


class GoodMemUploadError(ValueError):
    r"""Raised when a requested upload path is not allowed."""


def resolve_upload_path(path: str, upload_dir: Path | None) -> Path:
    r"""Resolves a model-supplied path inside the configured upload directory.

    Args:
        path (str): The path the caller or model supplied.
        upload_dir (Optional[Path]): The directory uploads are confined to.
            ``None`` means uploads were never enabled.

    Returns:
        Path: The resolved, allowed path.

    Raises:
        GoodMemUploadError: If uploads are not configured, if the path
            escapes the upload directory (directly, via ``..`` or through a
            symlink), or if it is not a readable regular file.
    """
    if upload_dir is None:
        raise GoodMemUploadError(
            "File uploads are disabled. Construct the toolkit with "
            "upload_dir=<directory> to enable them; only files inside that "
            "directory can be uploaded."
        )

    root = Path(upload_dir).expanduser().resolve(strict=False)
    # resolve() follows symlinks, so a link inside the directory that points
    # outside it is caught by the comparison below rather than followed.
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)

    if resolved != root and root not in resolved.parents:
        raise GoodMemUploadError(
            f"{path!r} is outside the upload directory {str(root)!r}."
        )
    if not resolved.exists():
        raise GoodMemUploadError(f"{path!r} does not exist.")
    if not resolved.is_file():
        raise GoodMemUploadError(f"{path!r} is not a regular file.")
    if not os.access(resolved, os.R_OK):
        raise GoodMemUploadError(f"{path!r} is not readable.")
    return resolved
