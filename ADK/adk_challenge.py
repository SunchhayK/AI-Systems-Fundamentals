"""
Google Agent Development Kit (ADK) framework challenge with local model support.

Implement the functions and classes marked with NotImplementedError according to
the contracts and constraints defined in FIELD_GUIDE.md.
"""

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.models import BaseLlm, LLMRegistry, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types


def get_exchange_rate(from_curr: str, to_curr: str) -> float:
    """
    Get the exchange rate between two currencies.

    Args:
        from_curr: Base currency code (e.g. 'USD').
        to_curr: Target currency code (e.g. 'EUR').

    Returns:
        Exchange rate float. For USD to EUR return 0.92, EUR to USD return 1.09,
        and identical currencies return 1.0. For other pairs, return 1.0.
    """
    raise NotImplementedError("TODO: implement get_exchange_rate")


def setup_currency_agent(model: str = "local-llama") -> Agent:
    """
    Construct a currency agent equipped with get_exchange_rate.

    Requirements:
    - Wrap get_exchange_rate with FunctionTool.
    - Instantiate and return an Agent with:
        name: 'currency_agent'
        model: model argument (defaults to 'local-llama')
        instruction: 'You assist users with currency exchange rates.'
        tools: list containing the wrapped FunctionTool.
    """
    raise NotImplementedError("TODO: implement setup_currency_agent")


def buy_asset(ticker: str, amount: int, tool_context: ToolContext) -> str:
    """
    Record an asset purchase in session state.

    Args:
        ticker: Asset symbol (e.g. 'GOOG', 'AAPL').
        amount: Number of shares/units purchased.
        tool_context: Active ADK tool context providing access to session state.

    Behavior:
    - Inspect tool_context.state for a 'holdings' dictionary. If missing, initialize it.
    - Increment the ticker count in 'holdings' by amount.
    - Return a confirmation string, e.g. f"Purchased {amount} shares of {ticker}".
    """
    raise NotImplementedError("TODO: implement buy_asset")


def setup_portfolio_agent(model: str = "local-llama") -> Agent:
    """
    Construct a portfolio management agent equipped with buy_asset.

    Requirements:
    - Wrap buy_asset with FunctionTool.
    - Instantiate and return an Agent with:
        name: 'portfolio_agent'
        model: model argument (defaults to 'local-llama')
        instruction: 'You manage investment portfolio assets and record purchases.'
        tools: list containing the wrapped FunctionTool.
    """
    raise NotImplementedError("TODO: implement setup_portfolio_agent")


def setup_financial_coordinator(model: str = "local-llama") -> Agent:
    """
    Construct a root coordinator agent that orchestrates the currency and portfolio specialists.

    Requirements:
    - Call setup_currency_agent(model=model) to get currency_agent.
    - Call setup_portfolio_agent(model=model) to get portfolio_agent.
    - Instantiate and return a root Agent with:
        name: 'financial_coordinator'
        model: model argument (defaults to 'local-llama')
        instruction: 'Coordinate financial questions and delegate to currency or portfolio specialists.'
        sub_agents: list containing currency_agent and portfolio_agent.
    """
    raise NotImplementedError("TODO: implement setup_financial_coordinator")


class LocalLlm(BaseLlm):
    """
    Custom local LLM provider integrating local models (Ollama, LM Studio, or local mock) into ADK.
    """

    model: str = "local-llama"

    @classmethod
    def supported_models(cls) -> list[str]:
        """
        Return regex patterns matching supported local model names.
        Should match names starting with 'local-', e.g. r'^local-.*'.
        """
        raise NotImplementedError("TODO: implement LocalLlm.supported_models")

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """
        Generate model response asynchronously.

        1. Extract user prompt text from llm_request.contents.
        2. Construct an answer string (either via local HTTP API like Ollama or local logic).
        3. Construct types.Content(role='model', parts=[types.Part.from_text(text=answer_text)]).
        4. Yield LlmResponse(content=content, partial=False).
        """
        raise NotImplementedError("TODO: implement LocalLlm.generate_content_async")
        yield  # Make it an async generator


def register_local_llm() -> None:
    """
    Register the LocalLlm class in the ADK LLMRegistry so agents can resolve 'local-.*' models.
    """
    raise NotImplementedError("TODO: implement register_local_llm")


def build_runner(agent: Agent, app_name: str = "financial_app") -> InMemoryRunner:
    """
    Construct an InMemoryRunner for the provided root agent.

    Args:
        agent: Root Agent instance.
        app_name: Name of the application.

    Returns:
        An initialized InMemoryRunner instance bound to agent.
    """
    raise NotImplementedError("TODO: implement build_runner")


def create_user_message(text: str) -> types.Content:
    """
    Construct a standard Google GenAI Content object representing a user turn.

    Args:
        text: User input prompt text.

    Returns:
        A types.Content object with role='user' and parts containing text.
    """
    raise NotImplementedError("TODO: implement create_user_message")
