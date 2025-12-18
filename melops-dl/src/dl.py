"""Minimal yt-dlp wrapper for extracting video info."""

from typing import Any

from yt_dlp import YoutubeDL


def download(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Download a single video and return its info dict.

    Uses extract_info with download=True to download and get metadata in one request.

    Args:
        url: Video URL supported by yt-dlp
        params: Dictionary of parameters for yt-dlp

    Returns:
        Sanitized info dict (JSON-serializable)
    """
    with YoutubeDL(params) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.sanitize_info(info)
