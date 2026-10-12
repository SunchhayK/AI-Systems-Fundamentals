import asyncio
import unittest
from unittest.mock import MagicMock

from google.adk.agents import Agent
from google.adk.models import LLMRegistry, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

try:
    import adk_challenge
except ModuleNotFoundError:
    from ADK import adk_challenge


class TestADKChallenge(unittest.TestCase):

    # --- MILESTONE 1: AGENT & STATELESS TOOL ---

    def test_milestone1_get_exchange_rate_logic(self):
        """Verify get_exchange_rate returns correct exchange rate floats."""
        usd_to_eur = adk_challenge.get_exchange_rate("USD", "EUR")
        self.assertAlmostEqual(usd_to_eur, 0.92, places=2)

        eur_to_usd = adk_challenge.get_exchange_rate("EUR", "USD")
        self.assertAlmostEqual(eur_to_usd, 1.09, places=2)

        same_curr = adk_challenge.get_exchange_rate("USD", "USD")
        self.assertEqual(same_curr, 1.0)

    def test_milestone1_currency_agent(self):
        """Verify currency_agent configuration and FunctionTool wrapping."""
        agent = adk_challenge.setup_currency_agent(model="local-llama")
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "currency_agent")
        self.assertEqual(agent.model, "local-llama")
        self.assertIn("currency", agent.instruction.lower())

        self.assertEqual(len(agent.tools), 1)
        tool = agent.tools[0]
        self.assertIsInstance(tool, FunctionTool)
        self.assertEqual(tool.name, "get_exchange_rate")

    # --- MILESTONE 2: STATEFUL TOOL & TOOLCONTEXT ---

    def test_milestone2_buy_asset_state_mutation(self):
        """Verify buy_asset records and accumulates asset holdings in session state."""
        mock_ctx = MagicMock(spec=ToolContext)
        mock_ctx.state = {}

        res1 = adk_challenge.buy_asset("GOOG", 10, mock_ctx)
        self.assertIn("10", res1)
        self.assertIn("GOOG", res1)
        self.assertIn("holdings", mock_ctx.state)
        self.assertEqual(mock_ctx.state["holdings"]["GOOG"], 10)

        res2 = adk_challenge.buy_asset("GOOG", 5, mock_ctx)
        self.assertIn("5", res2)
        self.assertEqual(mock_ctx.state["holdings"]["GOOG"], 15)

    def test_milestone2_portfolio_agent(self):
        """Verify portfolio_agent configuration and FunctionTool wrapping."""
        agent = adk_challenge.setup_portfolio_agent(model="local-llama")
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "portfolio_agent")
        self.assertEqual(agent.model, "local-llama")
        self.assertIn("portfolio", agent.instruction.lower())

        self.assertEqual(len(agent.tools), 1)
        tool = agent.tools[0]
        self.assertIsInstance(tool, FunctionTool)
        self.assertEqual(tool.name, "buy_asset")

    # --- MILESTONE 3: MULTI-AGENT HIERARCHY ---

    def test_milestone3_coordinator_hierarchy(self):
        """Verify financial_coordinator has both specialist sub-agents."""
        coordinator = adk_challenge.setup_financial_coordinator(model="local-llama")
        self.assertIsInstance(coordinator, Agent)
        self.assertEqual(coordinator.name, "financial_coordinator")
        self.assertEqual(coordinator.model, "local-llama")

        sub_agent_names = [sub.name for sub in coordinator.sub_agents]
        self.assertIn("currency_agent", sub_agent_names)
        self.assertIn("portfolio_agent", sub_agent_names)
        self.assertEqual(len(coordinator.sub_agents), 2)

        for sub in coordinator.sub_agents:
            self.assertEqual(sub.parent_agent, coordinator.name)

    # --- MILESTONE 4: CUSTOM LOCAL MODEL PROVIDER ---

    def test_milestone4_local_llm_registration(self):
        """Verify LocalLlm registers and resolves via LLMRegistry."""
        adk_challenge.register_local_llm()
        resolved_cls = LLMRegistry.resolve("local-llama")
        self.assertEqual(resolved_cls, adk_challenge.LocalLlm)

    def test_milestone4_local_llm_generation(self):
        """Verify LocalLlm.generate_content_async yields an LlmResponse with content."""
        llm_instance = adk_challenge.LocalLlm(model="local-llama")
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="What is the price of GOOG?")]
        )
        request = LlmRequest(model="local-llama", contents=[content])

        async def run_gen():
            responses = []
            async for resp in llm_instance.generate_content_async(request):
                responses.append(resp)
            return responses

        responses = asyncio.run(run_gen())
        self.assertGreater(len(responses), 0)
        self.assertIsInstance(responses[0], LlmResponse)
        self.assertFalse(responses[0].partial)
        self.assertIsNotNone(responses[0].content)

    # --- MILESTONE 5: LOCAL EXECUTION WITH INMEMORYRUNNER ---

    def test_milestone5_build_runner(self):
        """Verify build_runner returns an InMemoryRunner bound to the provided agent."""
        test_agent = Agent(name="test_bot", model="local-llama")
        runner = adk_challenge.build_runner(test_agent, app_name="test_app")
        self.assertIsInstance(runner, InMemoryRunner)
        self.assertEqual(runner.agent, test_agent)

    def test_milestone5_create_user_message(self):
        """Verify create_user_message produces a valid types.Content object with role 'user'."""
        msg = adk_challenge.create_user_message("What is the exchange rate?")
        self.assertIsInstance(msg, types.Content)
        self.assertEqual(msg.role, "user")
        self.assertGreater(len(msg.parts), 0)
        self.assertEqual(msg.parts[0].text, "What is the exchange rate?")

    def test_milestone5_runner_local_turn(self):
        """Verify full conversation turn executes locally using LocalLlm and InMemoryRunner."""
        adk_challenge.register_local_llm()
        agent = Agent(name="local_finance_bot", model="local-llama")
        runner = adk_challenge.build_runner(agent, app_name="finance_test")

        async def execute_turn():
            await runner.session_service.create_session(
                app_name="finance_test", user_id="u1", session_id="s1"
            )
            user_msg = adk_challenge.create_user_message("Hello from local test")
            return list(runner.run(user_id="u1", session_id="s1", new_message=user_msg))

        events = asyncio.run(execute_turn())
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].author, "local_finance_bot")


if __name__ == "__main__":
    unittest.main()
