# Project roadmap

Planned milestones for building AI system foundations from scratch.

## 1. RAG expansions

- Sparse retrieval: Pure Python BM25 scoring and inverted index.
- Hybrid search: Reciprocal rank fusion combining dense embeddings and BM25 scores.
- Reranking: Cross-encoder scoring and contextual compression.

## 2. Orchestration and LangChain internals

- Core chains: Composable pipeline runner for prompt, model call, and output parser.
- Tool calling: ReAct execution loop handling thought, action, observation, and finish states.
- Memory: Buffer and summary conversation stores.
- State machines: Graph-based state transitions with cyclic execution loops.

## 3. Multi-agent systems with Google ADK

- Communication protocol: Structured message bus between specialized agents.
- Topology patterns: Supervisor-worker orchestration and peer-to-peer handoffs.
- Tool sharing: Scoped tool registry and runtime permission checks.
- Google ADK integration: Agent definitions, session management, and runner execution.

## 4. Machine learning and deep learning architectures

- Autograd engine: Scalar reverse-mode automatic differentiation.
- Neural network layers: Dense layer, activation functions, and backpropagation in NumPy.
- Attention mechanisms: Scaled dot-product attention and multi-head attention.
- Transformer block: Layer normalization, residual additions, and causal self-attention.
