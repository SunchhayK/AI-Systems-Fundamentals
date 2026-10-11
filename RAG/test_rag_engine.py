import unittest

from rag_engine import RAGEngine, RetrievalResult


class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RAGEngine()

    # --- MILESTONE 1: CHUNKING & REPRESENTATION ---

    def test_milestone1_chunking_overlap(self):
        """Verify text chunking with overlap advances by (chunk_size - chunk_overlap)."""
        text = "abcdefghij"
        chunks = self.engine.chunk_text(text, chunk_size=4, chunk_overlap=2)
        expected = ["abcd", "cdef", "efgh", "ghij"]
        self.assertEqual(chunks, expected)

    def test_milestone1_chunking_no_overlap(self):
        """Verify chunking when overlap is zero."""
        text = "abcdef"
        chunks = self.engine.chunk_text(text, chunk_size=3, chunk_overlap=0)
        self.assertEqual(chunks, ["abc", "def"])

    def test_milestone1_chunking_short_and_empty(self):
        """Verify handling of text shorter than chunk_size and empty text."""
        self.assertEqual(
            self.engine.chunk_text("abc", chunk_size=10, chunk_overlap=2), ["abc"]
        )
        self.assertEqual(self.engine.chunk_text("", chunk_size=5, chunk_overlap=1), [])

    def test_milestone1_chunking_invalid_parameters(self):
        """Verify ValueError raised when parameters violate invariants."""
        with self.assertRaises(ValueError):
            self.engine.chunk_text("test", chunk_size=4, chunk_overlap=4)
        with self.assertRaises(ValueError):
            self.engine.chunk_text("test", chunk_size=4, chunk_overlap=5)
        with self.assertRaises(ValueError):
            self.engine.chunk_text("test", chunk_size=0, chunk_overlap=0)

    # --- MILESTONE 2: COSINE SIMILARITY ---

    def test_milestone2_cosine_identical(self):
        """Verify identical vectors yield similarity 1.0."""
        score = self.engine.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_milestone2_cosine_orthogonal(self):
        """Verify orthogonal vectors yield similarity 0.0."""
        score = self.engine.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_milestone2_cosine_opposite(self):
        """Verify opposite vectors yield similarity -1.0."""
        score = self.engine.cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        self.assertAlmostEqual(score, -1.0, places=5)

    def test_milestone2_cosine_zero_magnitude(self):
        """Verify zero vector safely returns 0.0 without division by zero error."""
        score = self.engine.cosine_similarity([0.0, 0.0], [1.0, 2.0])
        self.assertEqual(score, 0.0)

    def test_milestone2_cosine_dimension_mismatch(self):
        """Verify dimension mismatch raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    # --- MILESTONE 3: INDEXING & RETRIEVAL ---

    def test_milestone3_retrieval_ranking(self):
        """Verify retrieval scores chunks and returns sorted descending results."""
        docs = [
            {"id": "doc1", "text": "chunk one text"},
            {"id": "doc2", "text": "chunk two text"},
            {"id": "doc3", "text": "chunk three text"},
        ]

        def mock_embed(texts):
            mapping = {
                "chunk one text": [1.0, 0.0],
                "chunk two text": [0.7071, 0.7071],
                "chunk three text": [0.0, 1.0],
            }
            return [mapping.get(t, [0.0, 0.0]) for t in texts]

        count = self.engine.add_documents(
            docs, embedding_fn=mock_embed, chunk_size=50, chunk_overlap=0
        )
        self.assertEqual(count, 3)

        query_vec = [1.0, 0.0]
        results = self.engine.retrieve(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].doc_id, "doc1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)
        self.assertEqual(results[1].doc_id, "doc2")
        self.assertAlmostEqual(results[1].score, 0.7071, places=4)

    def test_milestone3_retrieval_empty_index(self):
        """Verify retrieving against an empty index returns an empty list."""
        results = self.engine.retrieve([1.0, 0.0], top_k=3)
        self.assertEqual(results, [])

    def test_milestone3_invalid_top_k(self):
        """Verify top_k <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.retrieve([1.0, 0.0], top_k=0)

    # --- MILESTONE 4: PROMPT AUGMENTATION ---

    def test_milestone4_prompt_assembly(self):
        """Verify augmented prompt contains retrieved context and query."""
        docs = [{"id": "d1", "text": "Antigravity is a coding assistant."}]

        def mock_embed(texts):
            return [[1.0, 0.0] for _ in texts]

        self.engine.add_documents(
            docs, embedding_fn=mock_embed, chunk_size=100, chunk_overlap=0
        )
        prompt = self.engine.build_augmented_prompt(
            query="What is Antigravity?",
            query_vector=[1.0, 0.0],
            top_k=1,
        )

        self.assertIn("Antigravity is a coding assistant.", prompt)
        self.assertIn("What is Antigravity?", prompt)

    def test_milestone4_prompt_assembly_empty_context(self):
        """Verify fallback when engine has no documents indexed."""
        prompt = self.engine.build_augmented_prompt(
            query="Empty query test",
            query_vector=[1.0, 0.0],
            top_k=2,
        )
        self.assertIn("Empty query test", prompt)


if __name__ == "__main__":
    unittest.main()
