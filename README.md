# AI Systems Fundamentals

Implementations built from scratch to study foundational AI system architectures.

## Topics

- [`RAG/`](RAG/): In-memory retrieval-augmented generation engine.
  - Character chunking with sliding window and overlap validation
  - Cosine similarity vector comparison using the standard library
  - In-memory document indexing and top-k retrieval
  - Augmented prompt assembly
- [`ADK/`](ADK/): Google Agent Development Kit (ADK) framework implementation.
  - Agent declaration and FunctionTool wrapping
  - Stateful session tools using ToolContext
  - Multi-agent hierarchy and coordinator delegation
  - Custom BaseLlm provider and LLMRegistry for local models (Ollama, LM Studio)
  - InMemoryRunner lifecycle and GenAI Content messaging

See [ROADMAP.md](ROADMAP.md) for upcoming modules.
