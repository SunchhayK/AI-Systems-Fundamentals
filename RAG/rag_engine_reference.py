"""
Reference implementation: In-memory RAG engine.

Demonstrates minimal, idiomatic standard library Python for:
- Deterministic text windowing without tail truncation.
- Vector cosine similarity with zero-norm guards and range clamping.
- Batch embedding generation to minimize network round-trips.
- Top-k selection using heapq.nlargest for O(N log k) retrieval.
- Context injection into parameterized prompt templates.
"""

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    vector: list[float] | None = None


@dataclass(slots=True)
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class RAGEngine:
    """
    Minimal in-memory Retrieval-Augmented Generation engine.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Split a string into overlapping character chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        if not text:
            return []

        step = chunk_size - chunk_overlap
        chunks = []
        for i in range(0, len(text), step):
            chunks.append(text[i : i + chunk_size])
            if i + chunk_size >= len(text):
                break
        return chunks

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute cosine similarity between two float vectors.
        Clamped to [-1.0, 1.0] to prevent floating point drift.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Vector lengths do not match")

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot / (norm_a * norm_b)
        # Clamp to guard against IEEE 754 precision drift (e.g. 1.0000000000000002)
        return max(-1.0, min(1.0, similarity))

    def add_documents(
        self,
        documents: list[dict],
        embedding_fn: Callable[[list[str]], list[list[float]]],
        chunk_size: int = 100,
        chunk_overlap: int = 20,
    ) -> int:
        """
        Chunk documents, batch-embed all chunks in a single call, and index them.
        Returns the number of new chunks added during this call.
        """
        # 1. Flatten all chunks across all documents while tracking parent IDs
        extracted_chunks: list[tuple[str, str]] = []  # (doc_id, chunk_text)
        for doc in documents:
            doc_id = doc["id"]
            for chunk_str in self.chunk_text(doc["text"], chunk_size, chunk_overlap):
                extracted_chunks.append((doc_id, chunk_str))

        if not extracted_chunks:
            return 0

        # 2. Batch-embed all chunk texts in a single network / compute pass
        all_texts = [text for _, text in extracted_chunks]
        embeddings = embedding_fn(all_texts)

        # 3. Store Chunk instances with globally unique compound IDs
        base_offset = len(self.chunks)
        new_chunks = [
            Chunk(
                chunk_id=f"{doc_id}_{base_offset + idx}",
                doc_id=doc_id,
                text=text,
                vector=vec,
            )
            for idx, ((doc_id, text), vec) in enumerate(
                zip(extracted_chunks, embeddings)
            )
        ]

        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def retrieve(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[RetrievalResult]:
        """
        Retrieve top_k most similar chunks using heapq.nlargest for O(N log k) speed.
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.chunks:
            return []

        def scored_chunks():
            for c in self.chunks:
                score = self.cosine_similarity(query_vector, c.vector)
                yield RetrievalResult(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    score=score,
                )

        # heapq.nlargest is O(N log k), faster than sorting all N elements (O(N log N))
        return heapq.nlargest(top_k, scored_chunks(), key=lambda r: r.score)

    def build_augmented_prompt(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 3,
        prompt_template: str = "Context:\n{context}\n\nQuestion: {query}\nAnswer:",
    ) -> str:
        """
        Retrieve context and format the prompt template for LLM inference.
        """
        results = self.retrieve(query_vector=query_vector, top_k=top_k)
        context = "\n\n".join(r.text for r in results)
        return prompt_template.format(context=context, query=query)


if __name__ == "__main__":
    # Demonstration of the reference engine
    engine = RAGEngine()

    corpus = [
        {
            "id": "arch_doc",
            "text": "RAG pairs vector similarity search with language model generation.",
        },
        {
            "id": "db_doc",
            "text": "PostgreSQL with pgvector enables persistent, indexed vector queries.",
        },
    ]

    def mock_embedder(texts: list[str]) -> list[list[float]]:
        # Toy embedding: assigns simple coordinates based on keywords
        vectors = []
        for t in texts:
            vec = [
                1.0 if "vector" in t.lower() else 0.1,
                1.0 if "rag" in t.lower() else 0.1,
            ]
            vectors.append(vec)
        return vectors

    indexed = engine.add_documents(
        corpus, embedding_fn=mock_embedder, chunk_size=40, chunk_overlap=10
    )
    print(f"Indexed {indexed} chunks across {len(corpus)} documents.")

    user_query = "How do vector databases help RAG?"
    query_vec = mock_embedder([user_query])[0]

    prompt = engine.build_augmented_prompt(
        query=user_query,
        query_vector=query_vec,
        top_k=2,
    )
    print("\n--- Assembled Prompt ---")
    print(prompt)
