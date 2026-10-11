# Field guide: Minimal in-memory RAG engine

## 1. Technology brief

- **Role and purpose**: Retrieval-Augmented Generation (RAG) grounds language model outputs on external reference text. It sits between an incoming query and an LLM prompt. Raw documents are split into chunks, converted into vector embeddings, indexed in memory, and queried via vector similarity to assemble a factual context window.
- **Core invariant**: Chunks preserve their parent document identifier, and similarity ranking remains monotonically sorted by cosine score descending.
- **State transitions**: `UNINITIALIZED` -> `INDEXED` (documents chunked, embedded, and stored) -> `RETRIEVED` (query vector scored against stored vectors) -> `AUGMENTED` (retrieved chunks formatted into prompt context).
- **Common failure modes**: Division by zero during cosine similarity calculation on zero-norm vectors, chunk overlap larger than chunk size, off-by-one boundary truncation in text slicing, and empty document collections.

---

## 2. Step-by-step implementation recipe

Follow these sequential milestones. Write the logic yourself using the contracts and invariants described.

### Milestone 1: Document chunking and data representation

- **What to do**: Implement `chunk_text(text, chunk_size, chunk_overlap)`. Split input text into overlapping windows of characters or tokens. Enforce validation: `chunk_size > 0`, `chunk_overlap >= 0`, and `chunk_overlap < chunk_size`. Return a list of string chunks.
- **Invariants to enforce**: Every slice advances by `step = chunk_size - chunk_overlap`. No chunk exceeds `chunk_size`. When remaining text is shorter than `chunk_size`, capture the final slice without duplicate loops.

### Milestone 2: Vector similarity calculation

- **What to do**: Implement `cosine_similarity(vec_a, vec_b)`. Calculate the dot product divided by the product of Euclidean norms using Python standard library `math` (`math.sqrt`).
- **Invariants to enforce**: If either vector has a Euclidean norm of zero (all zeros), return `0.0`. If lengths mismatch, raise `ValueError`. Return a float in the range `[-1.0, 1.0]`.

### Milestone 3: Document indexing and top-k retrieval

- **What to do**: Implement `add_documents(documents, embedding_fn)` and `retrieve(query_vector, top_k)`. Chunk each document, obtain embeddings using the provided callable, and store the resulting `Chunk` objects in memory. For retrieval, score each stored chunk against `query_vector`, sort descending, and return the top `k` matches as `RetrievalResult` objects.
- **Invariants to enforce**: `top_k` must be greater than zero. If the index is empty, return an empty list. Results must be sorted strictly descending by similarity score. Ties retain stable insertion order.

### Milestone 4: Prompt augmentation

- **What to do**: Implement `build_augmented_prompt(query, query_vector, top_k, prompt_template)`. Retrieve the top `k` chunks and substitute `{context}` and `{query}` into the template string.
- **Invariants to enforce**: If no documents exist or no chunks match, provide an empty or fallback context string without crashing. The returned string contains both query text and retrieved text blocks.

---

## 3. Individual component testing and verification

Test each milestone in isolation before moving to the next.

### Milestone 1 verification: Text chunking

- **Test runner command**:
  ```bash
  python3 -m unittest test_rag_engine.TestRAGEngine.test_milestone1_chunking_overlap
  ```
- **REPL check**:
  ```python
  from rag_engine import RAGEngine
  engine = RAGEngine()
  chunks = engine.chunk_text("abcdefghij", chunk_size=4, chunk_overlap=2)
  print(chunks)
  # Expected: ['abcd', 'cdef', 'efgh', 'ghij']
  ```
- **Expected output**:
  - Return value: List of string slices respecting stride length.
  - State mutation: None (pure function).
  - Error signature: `ValueError` when `chunk_overlap >= chunk_size`.

### Milestone 2 verification: Cosine similarity

- **Test runner command**:
  ```bash
  python3 -m unittest test_rag_engine.TestRAGEngine.test_milestone2_cosine_orthogonal
  ```
- **REPL check**:
  ```python
  from rag_engine import RAGEngine
  engine = RAGEngine()
  print(engine.cosine_similarity([1.0, 0.0], [0.0, 1.0]))
  # Expected: 0.0
  print(engine.cosine_similarity([0.0, 0.0], [1.0, 2.0]))
  # Expected: 0.0
  ```
- **Expected output**:
  - Return value: Float matching exact geometric similarity.
  - State mutation: None.
  - Error signature: `ValueError` when vector lengths differ.

### Milestone 3 verification: Indexing and retrieval

- **Test runner command**:
  ```bash
  python3 -m unittest test_rag_engine.TestRAGEngine.test_milestone3_retrieval_ranking
  ```
- **State inspection**:
  ```python
  from rag_engine import RAGEngine
  engine = RAGEngine()
  docs = [{"id": "doc1", "text": "Python is a programming language."}]
  engine.add_documents(docs, embedding_fn=lambda texts: [[1.0, 0.0]] * len(texts))
  print(len(engine.chunks))
  # Expected: Count of indexed chunks > 0
  ```
- **Expected output**:
  - Return value: List of `RetrievalResult` objects sorted by score descending.
  - State mutation: `engine.chunks` contains all embedded chunk records.

### Milestone 4 verification: Prompt augmentation

- **Test runner command**:
  ```bash
  python3 -m unittest test_rag_engine.TestRAGEngine.test_milestone4_prompt_assembly
  ```
- **Expected output**:
  - Return value: Formatted string containing context block and user query.
  - Boundary: Empty index produces template with empty context block.

---

## 4. Curated reference documentation

Consult these primary documentation sources while implementing:

### Standard library and core APIs

- [Python math.sqrt documentation](https://docs.python.org/3/library/math.html#math.sqrt)
  Consult for calculating Euclidean vector norms in standard library Python.
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
  Consult for structuring clean, typed data models without custom boilerplate.

### Mathematical foundation

- [Cosine Similarity on Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
  Consult for dot product normalization formula and range constraints.

### Isolated toy syntax example

This snippet demonstrates vector dot product and magnitude calculation in pure Python standard library:

```python
import math

vec_a = [1.0, 2.0, 3.0]
vec_b = [4.0, 5.0, 6.0]

dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
norm_a = math.sqrt(sum(a * a for a in vec_a))
norm_b = math.sqrt(sum(b * b for b in vec_b))
similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
```
