"""Small client for Spark Studio's SearXNG/DuckDuckGo search pipeline."""

from __future__ import annotations

from typing import Any

import httpx


def search(
    studio_url: str,
    query: str,
    *,
    limit: int = 5,
    enrich: bool = False,
    timeout: float = 45,
) -> dict[str, Any]:
    """Search through Spark Studio and return its normalized result payload."""
    query = query.strip()
    if not query:
        raise ValueError("search query cannot be empty")
    if len(query) > 500:
        raise ValueError("search query cannot exceed 500 characters")
    limit = max(1, min(int(limit), 10))
    base = studio_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("Spark Studio URL must start with http:// or https://")

    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{base}/api/search",
            params={"q": query, "limit": limit, "enrich": str(bool(enrich)).lower()},
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("Spark Studio returned an invalid search response")
    return payload


def format_results(payload: dict[str, Any]) -> str:
    """Render a compact, source-forward terminal result list."""
    backend = payload.get("url") or "unknown"
    lines = [f"Search: {payload.get('query') or ''}", f"Backend: {backend}"]
    results = payload.get("results") or []
    if not results:
        lines.append("No results.")
        return "\n".join(lines)
    for index, item in enumerate(results, 1):
        title = str(item.get("title") or item.get("url") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        summary = str(item.get("content") or item.get("snippet") or "").strip()
        summary = " ".join(summary.split())
        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."
        lines.extend(["", f"{index}. {title}", f"   {url}"])
        if summary:
            lines.append(f"   {summary}")
    return "\n".join(lines)
