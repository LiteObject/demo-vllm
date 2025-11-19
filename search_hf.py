"""Simple CLI helper to search Hugging Face models.

This script uses the Hugging Face Hub API to search for models by keyword,
optionally filtered by task (pipeline tag) and library.

Examples
--------
Search for Phi-3 text-generation models:
    python search_hf.py --query phi-3 --task text-generation --limit 10

Search for small chat models using Transformers:
    python search_hf.py --query chat --task text-generation --library transformers --limit 20
"""

from __future__ import annotations

import argparse
from typing import Iterable

from huggingface_hub import HfApi, ModelInfo


def _format_model(m: ModelInfo) -> str:
    """Return a one-line string describing a model search result."""
    model_id = getattr(m, "modelId", getattr(m, "id", "<unknown>"))
    tags = m.tags or []
    pipeline_tag = getattr(m, "pipeline_tag", None)
    task = pipeline_tag or next((t for t in tags if not t.startswith("library:")), "-")
    libs = [t.split(":", maxsplit=1)[1] for t in tags if t.startswith("library:")]
    libs_str = ",".join(libs) if libs else "-"
    likes = getattr(m, "likes", None)
    likes_str = f" ❤ {likes}" if likes is not None else ""

    downloads = getattr(m, "downloads", None)
    downloads_str = f" ⬇ {downloads}" if downloads is not None else ""

    last_modified = getattr(m, "lastModified", None) or getattr(
        m, "last_modified", None
    )
    last_modified_str = f" ⏱ {last_modified}" if last_modified is not None else ""

    return (
        f"{model_id:60}  task={task:22}  libs={libs_str}"
        f"{likes_str}{downloads_str}{last_modified_str}"
    )


def search_models(
    query: str, task: str | None, library: str | None, limit: int
) -> Iterable[ModelInfo]:
    """Search Hugging Face Hub for models matching the given filters."""
    api = HfApi()
    return api.list_models(
        search=query or None,
        task=task or None,
        library=library or None,
        limit=limit,
        sort="likes",
        cardData=False,
        fetch_config=False,
        full=True,
    )


def main() -> None:
    """Parse CLI arguments and print matching Hugging Face models."""

    parser = argparse.ArgumentParser(description="Search Hugging Face models")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Free-text search query (e.g. 'phi-3', 'qwen')",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Optional pipeline task filter (e.g. 'text-generation')",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="",
        help="Optional library filter (e.g. 'transformers', 'vllm')",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Maximum number of models to return"
    )

    args = parser.parse_args()

    models = list(search_models(args.query, args.task, args.library, args.limit))

    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} model(s):")
    for m in models:
        print("-", _format_model(m))


if __name__ == "__main__":
    main()
