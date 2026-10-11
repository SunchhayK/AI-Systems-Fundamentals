"""
RAG Engine starter stub.

Implement the methods marked with NotImplementedError according to the
contracts and constraints defined in FIELD_GUIDE.md.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    vector: list[float] | None = None


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float


def mock_embed(texts: list[str]) -> list[list[float]]:
    # Returns a 2D vector for each text
    return [[float(len(t)), 1.0] for t in texts]


class RAGEngine:
    """
    Minimal in-memory Retrieval-Augmented Generation engine.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """
        Split a string into overlapping chunks.

        Args:
            text: Input string to chunk.
            chunk_size: Maximum character length of each chunk. Must be > 0.
            chunk_overlap: Number of characters shared between adjacent chunks.
                           Must be >= 0 and < chunk_size.

        Returns:
            List of string chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        if len(text) == 0:
            return []

        if len(text) < chunk_size:
            return [text]

        chunks = []
        step = chunk_size - chunk_overlap

        for i in range(0, len(text), step):
            chunks.append(text[i : i + chunk_size])
            if i + chunk_size >= len(text):
                break

        return chunks

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute the cosine similarity between two float vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score between -1.0 and 1.0.
            Returns 0.0 if either vector has zero magnitude.

        Raises:
            ValueError: If vector lengths do not match.
        """

        if len(vec_a) != len(vec_b):
            raise ValueError("Vector lengths do not match")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

        return max(-1.0, min(1.0, similarity))

    def add_documents(
        self,
        documents: list[dict],
        embedding_fn: Callable[[list[str]], list[list[float]]],
        chunk_size: int = 100,
        chunk_overlap: int = 20,
    ) -> int:
        """
        Chunk documents, generate embeddings, and store them in memory.

        Args:
            documents: List of dicts with 'id' and 'text' keys.
            embedding_fn: Callable taking a list of chunk texts and returning
                          a list of float vectors.
            chunk_size: Character length for each chunk.
            chunk_overlap: Overlap length between adjacent chunks.

        Returns:
            Total number of chunks added.
        """
        for i, doc in enumerate(documents):
            chunk_texts = self.chunk_text(
                text=doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            embeddings = embedding_fn(chunk_texts)

            for i, text in enumerate(chunk_texts):
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{doc['id']}_{i}",
                        doc_id=doc["id"],
                        text=text,
                        vector=embeddings[i],
                    )
                )

        return len(self.chunks)

    def retrieve(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[RetrievalResult]:
        """
        Retrieve top_k chunks most similar to query_vector using cosine similarity.

        Args:
            query_vector: Embedding vector for the search query.
            top_k: Number of highest-scoring results to return. Must be > 0.

        Returns:
            List of RetrievalResult objects sorted descending by score.
        """
        if top_k <= 0:
            raise ValueError("Top K must be greater than 0.")

        if len(self.chunks) == 0:
            return []

        scores: list[RetrievalResult] = []
        for chunk in self.chunks:
            similarity = self.cosine_similarity(query_vector, chunk.vector)
            scores.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=similarity,
                    text=chunk.text,
                )
            )
        result = sorted(scores, key=lambda score: score.score, reverse=True)
        return result[:top_k]

    def build_augmented_prompt(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 3,
        prompt_template: str = "Context:\n{context}\n\nQuestion: {query}\nAnswer:",
    ) -> str:
        """
        Retrieve top_k chunks and assemble the final augmented prompt string.

        Args:
            query: User question string.
            query_vector: Embedding vector for the user query.
            top_k: Number of context chunks to retrieve.
            prompt_template: Template string containing {context} and {query} placeholders.

        Returns:
            Formatted prompt string ready for LLM inference.
        """

        results = self.retrieve(query_vector=query_vector, top_k=top_k)

        context_str = "\n\n".join(r.text for r in results)

        return prompt_template.format(
            context=context_str,
            query=query,
        )


# text_25 = "abcdefghijklmnopqrstuvwxy"
# text_10 = "abcdefghijk"


rag = RAGEngine()

# print(rag.chunk_text(text=text_25, chunk_size=25, chunk_overlap=20))

# vec1 = [1.0, 2.0, 3.0]
# vec2 = [2.0, 2.0, 3.0]

# print(rag.cosine_similarity(vec1, vec2))

docs = [
    {"id": "doc1", "text": "Antigravity is a coding document."},
    {"id": "doc2", "text": "chunk two text"},
    {"id": "doc3", "text": "chunk three text"},
    {
        "id": "doc4",
        "text": "abcdefghijklmnopqrstuvwxyz0123456789",  # 36 characters
    },
]

count = rag.add_documents(documents=docs, embedding_fn=mock_embed)

query = "What is Antigravity?"
query_vec = mock_embed(rag.chunk_text(query, chunk_overlap=0, chunk_size=100))[0]
prompt = rag.build_augmented_prompt(
    query=query,
    query_vector=query_vec,
    top_k=3,
)

print(prompt)
