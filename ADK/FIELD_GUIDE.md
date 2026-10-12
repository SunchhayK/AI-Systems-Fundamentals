# Field guide: Google Agent Development Kit (ADK) with local models

## 1. Technology brief

- **Role and purpose**: The Google Agent Development Kit (ADK) is an open-source, code-first Python framework for building and orchestrating AI agents. While optimized for Gemini, ADK includes an extensible model subsystem (`BaseLlm` and `LLMRegistry`). This architecture allows developers to route agent reasoning entirely to local models (such as Ollama, llama.cpp, LM Studio, or local mock runtimes) without paying for API calls or relying on cloud networks.
- **Core invariant**: Models inherit from `BaseLlm` and register matching regex patterns in `LLMRegistry`. Agents instantiate models by name, and state mutations persist across session turns via `ToolContext`.
- **State transitions**: `MODEL_REGISTERED` (local provider added to registry) -> `AGENT_INITIALIZED` (tools wrapped in `FunctionTool`) -> `RUNNER_MOUNTED` (bound to `InMemoryRunner`) -> `LOCAL_INFERENCE` (`generate_content_async` dispatches to local model or endpoint) -> `EVENT_STREAMED` (response emitted).
- **Common failure modes**: Missing `supported_models()` regex match during model resolution, forgetting to create a session before calling `runner.run()`, unhandled connection errors when the local Ollama/model server is offline, and omitting `partial=False` on the final `LlmResponse`.

---

## 2. Step-by-step implementation recipe

Follow these milestones in `adk_challenge.py` to construct a fully local, self-contained Google ADK agent system.

### Milestone 1: Agent declaration and stateless tools

- **What to do**: Implement `get_exchange_rate(from_curr: str, to_curr: str) -> float` and `setup_currency_agent(model="local-llama") -> Agent`. Wrap the function using `FunctionTool(get_exchange_rate)`. Configure the agent with `name="currency_agent"`, the specified model identifier, and clear instructions.
- **Invariants to enforce**: Tools in `agent.tools` must be instances of `FunctionTool`. The model argument must default to `"local-llama"`.

### Milestone 2: Stateful tools with ToolContext

- **What to do**: Implement `buy_asset(ticker: str, amount: int, tool_context: ToolContext) -> str` and `setup_portfolio_agent(model="local-llama") -> Agent`. The tool checks `tool_context.state` for a `"holdings"` dictionary, initializes it if missing, increments the ticker count, and returns a confirmation string.
- **Invariants to enforce**: The tool signature must accept `tool_context: ToolContext`. State modifications must directly mutate `tool_context.state["holdings"]`.

### Milestone 3: Multi-agent hierarchy and coordinator delegation

- **What to do**: Implement `setup_financial_coordinator(model="local-llama") -> Agent`. Instantiate `currency_agent` and `portfolio_agent`, then attach them as `sub_agents` to a parent agent named `"financial_coordinator"`.
- **Invariants to enforce**: Each sub-agent belongs to only one parent agent. ADK validates agent trees at initialization.

### Milestone 4: Custom local model provider

- **What to do**: Implement `LocalLlm(BaseLlm)` and `register_local_llm()`. Declare `supported_models` returning `[r"^local-.*"]`. Implement the asynchronous generator `generate_content_async(llm_request, stream=False)`. Extract user text from `llm_request.contents`, dispatch to your local endpoint (or generate a local response), and yield an `LlmResponse(content=content, partial=False)`. Register the class via `LLMRegistry.register(LocalLlm)`.
- **Invariants to enforce**: `LLMRegistry.resolve("local-llama")` must return `LocalLlm`. The generator must yield at least one `LlmResponse` with `partial=False`.

### Milestone 5: Local execution with InMemoryRunner

- **What to do**: Implement `build_runner(agent: Agent) -> InMemoryRunner` and `create_user_message(text: str) -> types.Content`. Verify that `runner.run()` processes turns using your local model without making external API calls.
- **Invariants to enforce**: Messages must use standard `google.genai.types.Content` with role `"user"`.

---

## 3. Individual component testing and verification

Test each milestone in isolation using the test runner or interactive REPL checks.

### Milestone 1 verification: Stateless tool and agent

- **Test runner command**:
  ```bash
  ADK/.venv/bin/python -m unittest test_adk_challenge.TestADKChallenge.test_milestone1_get_exchange_rate_logic \
                                  test_adk_challenge.TestADKChallenge.test_milestone1_currency_agent
  ```

### Milestone 2 verification: Stateful tool with ToolContext

- **Test runner command**:
  ```bash
  ADK/.venv/bin/python -m unittest test_adk_challenge.TestADKChallenge.test_milestone2_buy_asset_state_mutation \
                                  test_adk_challenge.TestADKChallenge.test_milestone2_portfolio_agent
  ```

### Milestone 3 verification: Multi-agent hierarchy

- **Test runner command**:
  ```bash
  ADK/.venv/bin/python -m unittest test_adk_challenge.TestADKChallenge.test_milestone3_coordinator_hierarchy
  ```

### Milestone 4 verification: Local model registration

- **Test runner command**:
  ```bash
  ADK/.venv/bin/python -m unittest test_adk_challenge.TestADKChallenge.test_milestone4_local_llm_registration \
                                  test_adk_challenge.TestADKChallenge.test_milestone4_local_llm_generation
  ```
- **REPL check**:
  ```python
  from adk_challenge import register_local_llm
  from google.adk.models import LLMRegistry
  register_local_llm()
  cls = LLMRegistry.resolve("local-llama")
  print("Resolved class:", cls.__name__)
  # Expected: LocalLlm
  ```

### Milestone 5 verification: Runner with local model turn

- **Test runner command**:
  ```bash
  ADK/.venv/bin/python -m unittest test_adk_challenge.TestADKChallenge.test_milestone5_runner_local_turn
  ```

---

## 4. Connecting a real local model server

### Option A: Connecting to local Ollama

If you run Ollama locally (`ollama run llama3.2` or `ollama run mistral`):

- Ollama provides a local HTTP API at `http://localhost:11434/api/generate` or `http://localhost:11434/api/chat`.
- Inside `LocalLlm.generate_content_async`, send an HTTP POST request to Ollama using `urllib.request` or `httpx`:

  ```python
  import json
  import urllib.request

  payload = json.dumps({"model": "llama3.2", "prompt": user_prompt, "stream": False}).encode("utf-8")
  req = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req) as response:
      res_data = json.loads(response.read().decode("utf-8"))
      model_text = res_data.get("response", "")
  ```

### Option B: Connecting to LM Studio or llama.cpp (OpenAI-compatible)

If you use LM Studio or llama.cpp:

- The local server exposes `http://localhost:1234/v1/chat/completions` or `http://localhost:8080/v1/chat/completions`.
- Inside `LocalLlm`, send standard OpenAI-format messages to the local endpoint and return the generated text.

### Option C: Google AI Studio free tier

If you want to test with Gemini directly later:

- Google AI Studio offers a developer tier with no billing required (up to 15 requests per minute free).
- Set `export GEMINI_API_KEY="your-key"` and pass `model="gemini-2.5-flash"`.

---

## 5. Curated reference documentation

- [Google ADK Python Repository](https://github.com/google/adk-python)
  Primary repository containing SDK modules (`google.adk.models.BaseLlm`, `LLMRegistry`).
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
  Specification for local HTTP generate and chat completions endpoints.
- [Google GenAI Python SDK Types](https://github.com/googleapis/python-genai)
  Reference for `types.Content` and `types.Part`.

### Isolated toy syntax example: Custom BaseLlm

```python
from typing import AsyncGenerator
from google.adk.models import BaseLlm, LLMRegistry, LlmRequest, LlmResponse
from google.genai import types

class EchoLlm(BaseLlm):
    model: str = "local-echo"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"^local-echo.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        text = "Local response"
        content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
        yield LlmResponse(content=content, partial=False)

LLMRegistry.register(EchoLlm)
```
