"""Minimal yt-dlp wrapper for extracting video info."""

from typing import Any

from yt_dlp import YoutubeDL


def download(urls: list[str], params: dict[str, Any]) -> None:
    """
    Download videos using yt-dlp with specified parameters.

    Args:
        urls: List of video URLs supported by yt-dlp
        params: Dictionary of parameters for yt-dlp
    """
    with YoutubeDL(params) as ydl:
        ydl.download(urls)
