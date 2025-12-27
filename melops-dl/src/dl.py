"""Minimal yt-dlp wrapper for extracting video info."""

from typing import Any

from yt_dlp import YoutubeDL


def download(url: str, params: Any) -> tuple[str | None, Any]:
    """
    Download a single video and return final file path and metadata.

    Uses extract_info with download=True and post_hooks to capture the final
    file path after all post-processing (FFmpeg conversion, file moves, etc.).

    Args:
        url: Video URL supported by yt-dlp
        params: Dictionary of parameters for yt-dlp

    Returns:
        Tuple of (file_path, sanitized_info_dict):
        - file_path: Absolute path to final downloaded file (None if download failed)
        - sanitized_info_dict: JSON-serializable metadata (id, title, duration, etc.)
    """
    file_path: str | None = None

    def post_hook(val: str) -> None:
        nonlocal file_path
        file_path = val

    # Add post hook to params
    params.setdefault("post_hooks", []).append(post_hook)

    with YoutubeDL(params) as ydl:
        info = ydl.extract_info(url, download=True)
        sanitized = ydl.sanitize_info(info)

        if sanitized is None:
            sanitized = info

        return file_path, sanitized
