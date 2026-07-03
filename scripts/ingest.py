"""
scripts/ingest.py

Command-line knowledge ingestion pipeline for VoiceReach AI.

Loads knowledge source files from a directory (JSON, TXT, Markdown, CSV),
converts each into one or more KnowledgeChunk objects, and inserts them
into the Chroma-backed KnowledgeRetriever via `add_documents()`.

Usage:
    python -m scripts.ingest --source data/knowledge
    python -m scripts.ingest --source data/knowledge --collection voice_reach_knowledge

Supported source formats
-------------------------
JSON (.json):
    Either a single object or a list of objects, each shaped like:
        {
            "id": "malaria_001",            (optional -- generated if absent)
            "title": "Malaria symptoms",
            "category": "malaria",
            "content": "...",
            "source": "NCDC Guidelines 2023",
            "tags": ["malaria", "symptoms"]  (optional)
        }

TXT / Markdown (.txt, .md):
    The whole file becomes a single KnowledgeChunk. `title` is derived
    from the filename, `category` defaults to "general", `source`
    defaults to the filename.

CSV (.csv):
    One row per KnowledgeChunk. Expected columns: title, category,
    content, source, and an optional tags column (semicolon-separated).
    An "id" column is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.models import KnowledgeChunk
from app.services.knowledge import KnowledgeRetriever
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    """Generate a stable-looking, collision-resistant chunk id."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _load_json_file(path: Path) -> list[KnowledgeChunk]:
    """Parse a .json file into one or more KnowledgeChunks."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]

    chunks: list[KnowledgeChunk] = []
    for record in records:
        chunks.append(
            KnowledgeChunk(
                id=record.get("id") or _generate_id(path.stem),
                title=record["title"],
                category=record.get("category", "general"),
                content=record["content"],
                source=record.get("source", path.name),
                tags=record.get("tags", []),
            )
        )
    return chunks


def _load_text_file(path: Path) -> list[KnowledgeChunk]:
    """Parse a .txt or .md file into a single KnowledgeChunk."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    return [
        KnowledgeChunk(
            id=_generate_id(path.stem),
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            category="general",
            content=content,
            source=path.name,
            tags=[],
        )
    ]


def _load_csv_file(path: Path) -> list[KnowledgeChunk]:
    """Parse a .csv file into one KnowledgeChunk per row."""
    chunks: list[KnowledgeChunk] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tags_raw = (row.get("tags") or "").strip()
            tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []
            chunks.append(
                KnowledgeChunk(
                    id=(row.get("id") or "").strip() or _generate_id(path.stem),
                    title=row["title"],
                    category=row.get("category", "general"),
                    content=row["content"],
                    source=row.get("source", path.name),
                    tags=tags,
                )
            )
    return chunks


_LOADERS: dict[str, Any] = {
    ".json": _load_json_file,
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".csv": _load_csv_file,
}


def load_knowledge_files(source_dir: Path) -> list[KnowledgeChunk]:
    """Walk `source_dir` and load every supported file into KnowledgeChunks.

    Args:
        source_dir: Directory containing knowledge source files.

    Returns:
        A flat list of all KnowledgeChunks parsed from every file found.
    """
    all_chunks: list[KnowledgeChunk] = []

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file type: %s", path)
            continue

        try:
            chunks = loader(path)
        except Exception as exc:  # noqa: BLE001 -- a bad source file must
            # not abort the whole ingestion run.
            logger.error("Failed to load %s: %s", path, exc)
            continue

        logger.info("Loaded %d chunk(s) from %s", len(chunks), path.name)
        all_chunks.extend(chunks)

    return all_chunks


def main() -> None:
    """CLI entry point: parse args, load files, ingest into Chroma, report."""
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Ingest knowledge files into VoiceReach AI.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/knowledge"),
        help="Directory containing .json/.txt/.md/.csv knowledge files.",
    )
    parser.add_argument(
        "--persist-directory",
        type=str,
        default=settings.chroma_persist_directory,
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.chroma_collection_name,
        help="Chroma collection name to ingest into.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        logger.error("Source directory does not exist: %s", args.source)
        sys.exit(1)

    chunks = load_knowledge_files(args.source)
    if not chunks:
        logger.warning("No knowledge chunks found under %s. Nothing to ingest.", args.source)
        sys.exit(0)

    retriever = KnowledgeRetriever(
        persist_directory=args.persist_directory,
        collection_name=args.collection,
        embedding_model=settings.embedding_model,
    )

    inserted = retriever.add_documents(chunks)

    print("\n--- Ingestion Summary ---")
    print(f"Source directory : {args.source}")
    print(f"Collection       : {args.collection}")
    print(f"Chunks parsed    : {len(chunks)}")
    print(f"Chunks inserted  : {len(inserted)}")
    print(f"Total in store   : {retriever.count_documents()}")
    print("-------------------------\n")


if __name__ == "__main__":
    main()
