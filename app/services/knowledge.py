"""Knowledge retrieval services for VoiceReach AI.

This module implements a production-quality Retrieval-Augmented Generation
layer backed by ChromaDB and sentence-transformers. It is intentionally
self-contained and compatible with the existing orchestrator contract:

    retrieve(query: str, top_k: int = 3) -> list[KnowledgeChunk]

The retriever persists data under data/chroma/ and exposes methods for
ingesting documents, retrieving semantically similar content, and inspecting
collection state without changing any existing public contracts elsewhere in
this backend.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.models import KnowledgeChunk

logger = logging.getLogger(__name__)


class _SentenceTransformerEmbeddingFunction:
    """Thin adapter that exposes the local embedder to ChromaDB."""

    def __init__(self, embedder: "KnowledgeRetriever") -> None:
        self._embedder = embedder

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Return dense embeddings for the supplied text values.
        
        Args:
            input: Sequence of text strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        return self._embedder.embed(list(input))

    def name(self) -> str:
        """Return the embedding function name for ChromaDB compatibility."""
        return "voice-reach-sentence-transformer"


class KnowledgeRetriever:
    """Persistent semantic retrieval layer for trusted healthcare/financial knowledge.

    The retriever initializes a persistent ChromaDB collection on disk,
    creates embeddings with sentence-transformers using the
    all-MiniLM-L6-v2 model, and exposes a simple interface for ingestion
    and retrieval that is compatible with the orchestrator.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "voice_reach_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the retriever and its backing storage.

        Args:
            persist_directory: Optional override for the Chroma persistence path.
                When omitted, data/chroma/ under the repository root is used.
            collection_name: Name of the Chroma collection to create or reuse.
            embedding_model: sentence-transformers model identifier.
        """
        self.persist_directory = Path(
            persist_directory or (Path(__file__).resolve().parents[2] / "data" / "chroma")
        ).resolve()
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_function: Optional[_SentenceTransformerEmbeddingFunction] = None

        self._initialize()

    def _initialize(self) -> None:
        """Create the persistence directory and initialize Chroma's collection."""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._embedding_function = _SentenceTransformerEmbeddingFunction(self)
            self._collection = self._create_collection()
            logger.info(
                "Knowledge retriever initialized for collection=%s at %s",
                self.collection_name,
                self.persist_directory,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Knowledge retriever initialization failed: %s", exc)
            self._client = None
            self._collection = None

    def _create_collection(self) -> Optional[Any]:
        """Create or reuse the backing Chroma collection for knowledge storage."""
        if self._client is None:
            return None

        try:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_function,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to create Chroma collection %s: %s", self.collection_name, exc)
            return None

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence-transformers embedding model on demand."""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            self._embedding_model = SentenceTransformer(self.embedding_model)
            return self._embedding_model
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to load embedding model %s: %s", self.embedding_model, exc)
            raise RuntimeError(f"Unable to load embedding model: {self.embedding_model}") from exc

    def embed(self, text: str | Sequence[str]) -> list[float] | list[list[float]]:
        """Generate embeddings for one or more text values.

        Args:
            text: A single string or a sequence of strings to embed.

        Returns:
            A single embedding vector for a single string, or a list of vectors
            for multiple strings.
        """
        if isinstance(text, str):
            texts = [text]
            is_single = True
        else:
            texts = [value for value in text if isinstance(value, str) and value.strip()]
            is_single = False

        if not texts:
            return [] if not is_single else []

        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if hasattr(embeddings, "tolist"):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = list(embeddings)

            if is_single:
                return embeddings_list[0]
            return embeddings_list
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Embedding generation failed: %s", exc)
            raise RuntimeError("Embedding generation failed") from exc

    def _coerce_document(self, document: KnowledgeChunk | dict[str, Any]) -> KnowledgeChunk:
        """Normalize a supplied document into the canonical KnowledgeChunk model."""
        if isinstance(document, KnowledgeChunk):
            chunk = document
        elif isinstance(document, dict):
            chunk = KnowledgeChunk(
                id=str(document.get("id") or str(uuid.uuid4())),
                title=str(document.get("title") or "Untitled"),
                category=str(document.get("category") or "general"),
                content=str(document.get("content") or ""),
                source=str(document.get("source") or "unknown"),
                tags=list(document.get("tags") or []),
            )
        else:
            raise TypeError("Knowledge documents must be KnowledgeChunk instances or dictionaries")

        if not chunk.id:
            chunk.id = str(uuid.uuid4())
        if not chunk.content.strip():
            raise ValueError("Knowledge content cannot be empty")

        return chunk

    def _metadata_from_chunk(self, chunk: KnowledgeChunk) -> dict[str, Any]:
        """Serialize a KnowledgeChunk into Chroma-compatible metadata."""
        return {
            "id": chunk.id,
            "title": chunk.title,
            "category": chunk.category,
            "source": chunk.source,
            "tags": ",".join(chunk.tags),
        }

    def add_document(self, document: KnowledgeChunk | dict[str, Any]) -> Optional[KnowledgeChunk]:
        """Add a single document to the knowledge collection.

        Duplicate IDs are skipped to prevent repeated ingestion. The method
        logs and returns None when the collection is unavailable.
        """
        try:
            chunk = self._coerce_document(document)
        except (TypeError, ValueError) as exc:
            logger.warning("Rejected invalid document: %s", exc)
            return None

        if self._collection is None:
            logger.warning("Knowledge collection is not available; document skipped")
            return None

        try:
            existing = self._collection.get(ids=[chunk.id])
            if existing and existing.get("ids"):
                logger.info("Document %s already exists; skipping", chunk.id)
                return chunk

            self._collection.add(
                ids=[chunk.id],
                documents=[chunk.content],
                metadatas=[self._metadata_from_chunk(chunk)],
            )
            logger.info("Added document %s to knowledge collection", chunk.id)
            return chunk
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to add document %s: %s", chunk.id, exc)
            return None

    def add_documents(self, documents: Sequence[KnowledgeChunk | dict[str, Any]]) -> list[KnowledgeChunk]:
        """Batch ingest multiple documents while preventing duplicates."""
        if self._collection is None:
            logger.warning("Knowledge collection is not available; batch skipped")
            return []

        normalized_chunks: list[KnowledgeChunk] = []
        for document in documents:
            try:
                normalized_chunks.append(self._coerce_document(document))
            except (TypeError, ValueError) as exc:
                logger.warning("Rejected invalid document during batch ingest: %s", exc)

        if not normalized_chunks:
            return []

        ids = [chunk.id for chunk in normalized_chunks]
        try:
            existing_result = self._collection.get(ids=ids)
            existing_ids = set(existing_result.get("ids", [])) if existing_result else set()
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to inspect existing documents during batch ingest: %s", exc)
            existing_ids = set()

        chunks_to_add = [chunk for chunk in normalized_chunks if chunk.id not in existing_ids]
        if not chunks_to_add:
            logger.info("All %s documents already existed; nothing to add", len(normalized_chunks))
            return [chunk for chunk in normalized_chunks if chunk.id in existing_ids]

        try:
            self._collection.add(
                ids=[chunk.id for chunk in chunks_to_add],
                documents=[chunk.content for chunk in chunks_to_add],
                metadatas=[self._metadata_from_chunk(chunk) for chunk in chunks_to_add],
            )
            logger.info("Added %s documents to knowledge collection", len(chunks_to_add))
            return chunks_to_add
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to add batch of documents: %s", exc)
            return []

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """Retrieve the most semantically relevant knowledge chunks for a query.

        The method uses cosine similarity via Chroma's query API and converts
        the distance returned by Chroma into a similarity score in the range
        $[0, 1]$.
        """
        if not query or not query.strip():
            return []
        if self._collection is None:
            logger.warning("Knowledge collection is not available; retrieval skipped")
            return []

        try:
            query_embedding = self.embed(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Knowledge retrieval failed for query=%r: %s", query, exc)
            return []

        chunks: list[KnowledgeChunk] = []
        try:
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            for document_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                if not document:
                    continue
                metadata_dict = metadata or {}
                score = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
                tags = []
                if isinstance(metadata_dict.get("tags"), str):
                    tags = [tag.strip() for tag in metadata_dict["tags"].split(",") if tag.strip()]
                chunks.append(
                    KnowledgeChunk(
                        id=str(document_id),
                        title=str(metadata_dict.get("title") or "Untitled"),
                        category=str(metadata_dict.get("category") or "general"),
                        content=str(document),
                        source=str(metadata_dict.get("source") or "unknown"),
                        tags=tags,
                        score=round(score, 6),
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to parse retrieval results: %s", exc)
            return []

        return chunks

    def count_documents(self) -> int:
        """Return the number of documents currently stored in the collection."""
        if self._collection is None:
            return 0

        try:
            return int(self._collection.count())
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to count documents: %s", exc)
            return 0

    def delete_document(self, document_id: str) -> None:
        """Delete a single document from the collection by its identifier."""
        if not document_id:
            return
        if self._collection is None:
            logger.warning("Knowledge collection is not available; delete skipped")
            return

        try:
            self._collection.delete(ids=[document_id])
            logger.info("Deleted document %s from knowledge collection", document_id)
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to delete document %s: %s", document_id, exc)

    def clear_collection(self) -> None:
        """Remove all documents from the collection and recreate it."""
        if self._client is None:
            return

        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive runtime handling
            logger.exception("Unable to delete collection %s: %s", self.collection_name, exc)

        self._collection = self._create_collection()
        logger.info("Cleared knowledge collection %s", self.collection_name)

    def collection_stats(self) -> dict[str, Any]:
        """Return a lightweight summary of the collection state."""
        return {
            "collection": self.collection_name,
            "count": self.count_documents(),
            "persist_directory": str(self.persist_directory),
            "available": self._collection is not None,
        }
