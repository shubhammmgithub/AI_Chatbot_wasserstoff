import logging
import uuid
from typing import List, Dict, Any

from sklearn.cluster import KMeans
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from backend.app.core.logger import setup_logger
from backend.app.core.config import get_qdrant_client, EMBED_MODEL_NAME, FALLBACK_DIM
from backend.app.utils.exceptions import UpsertError

logger = setup_logger("embedding_service")

class EmbeddingService:
    """Service for embedding texts and upserting them into session-specific Qdrant collections."""

    def __init__(self):
        """Initialize the embedding service with the model and Qdrant client."""
        self.embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.qdrant = get_qdrant_client()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts into vectors."""
        if not texts: return []
        logger.info(f"Embedding {len(texts)} texts...")
        return self.embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def _assign_themes(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], n_clusters: int = 5) -> List[str]:
        """Clusters embeddings into themes using KMeans."""
        n_chunks = len(chunks)
        if n_chunks < n_clusters:
            # If there are fewer chunks than desired clusters, assign each to its own theme
            return [f"Theme-{i}" for i in range(n_chunks)]
        
        logger.info(f"Clustering {n_chunks} chunks into {n_clusters} themes.")
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(embeddings)
        return [f"Theme-{label}" for label in labels]

    def _ensure_collection(self, collection_name: str, vector_size: int):
        """Creates a Qdrant collection if it doesn't already exist."""
        try:
            self.qdrant.get_collection(collection_name=collection_name)
        except Exception:
            logger.info(f"Creating new session collection in Qdrant: {collection_name}")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]], session_id: str) -> int:
        """
        Embeds, clusters, and upserts chunks into a session-specific Qdrant collection.
        """
        if not chunks:
            logger.warning("No chunks provided to upsert.")
            return 0

        # 1. Define the unique collection name for this session
        collection_name = f"session_{session_id}"
        
        # 2. Embed all text chunks
        texts = [c.get("text", "") for c in chunks]
        embeddings = self._embed_texts(texts)
        
        # 3. Ensure the session's collection exists
        vector_size = len(embeddings[0]) if embeddings else FALLBACK_DIM
        self._ensure_collection(collection_name, vector_size)
        
        # 4. Assign themes to the chunks
        themes = self._assign_themes(chunks, embeddings)
        
        # 5. Prepare points for upserting
        points_batch = []
        for chunk, vector, theme in zip(chunks, embeddings, themes):
            payload = {
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "para": chunk.get("para"),
                "text": chunk.get("text"),
                "theme": theme,
            }
            points_batch.append(
                qmodels.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            )

        # 6. Upsert points to the session-specific collection
        try:
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points_batch,
                wait=True
            )
            logger.info(f"Successfully upserted {len(points_batch)} chunks to collection '{collection_name}'")
            return len(points_batch)
        except Exception as e:
            logger.exception(f"Failed to upsert chunks to collection '{collection_name}': {e}")
            raise UpsertError(f"Failed to save data for session {session_id}")