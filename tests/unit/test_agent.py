"""Unit tests for the agent loop. The LLM and the tools are fakes."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from chatbot.core.agent import run_agent
from chatbot.core.llm.gemini import LLMError
from chatbot.core.tools.web_search import SearchResult, ToolError


def _text_response(text: str) -> SimpleNamespace:
    """Build a fake model response carrying final text."""
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(
        text=text, function_calls=None, candidates=[SimpleNamespace(content=content)]
    )


def _call_response(*calls: tuple[str, dict[str, Any]]) -> SimpleNamespace:
    """Build a fake model response requesting the given function calls."""
    function_calls = [types.FunctionCall(name=name, args=args) for name, args in calls]
    parts = [types.Part(function_call=call) for call in function_calls]
    content = types.Content(role="model", parts=parts)
    return SimpleNamespace(
        text=None, function_calls=function_calls, candidates=[SimpleNamespace(content=content)]
    )


def _gemini(*responses: SimpleNamespace) -> AsyncMock:
    """Build a fake GeminiClient replaying the given responses in order."""
    gemini = AsyncMock()
    gemini.generate = AsyncMock(side_effect=list(responses))
    return gemini


def _results(*urls: str) -> list[SearchResult]:
    """Build search results pointing at the given urls."""
    return [SearchResult(title=f"t{i}", url=url, description=f"d{i}") for i, url in enumerate(urls)]


async def _run(gemini: AsyncMock, tools: dict[str, Any], **kwargs: Any) -> str:
    """Run the agent with test-friendly defaults."""
    defaults: dict[str, Any] = {"max_iterations": 4, "deadline_seconds": 90}
    return await run_agent(
        gemini,
        subject="S",
        body="B",
        attachments=[],
        thread=[],
        tools=tools,
        **{**defaults, **kwargs},
    )


class TestPassthrough:
    """Tests for the no-tool path."""

    async def test_direct_answer_without_tools(self) -> None:
        """Verify an empty tool set means one plain model call."""
        gemini = _gemini(_text_response("Hello"))

        reply = await _run(gemini, {})

        assert reply == "Hello"
        assert gemini.generate.await_count == 1
        assert gemini.generate.await_args.kwargs["tools"] is None

    async def test_direct_answer_with_tools_offered(self) -> None:
        """Verify a declared tool set is offered even when the model answers directly."""
        gemini = _gemini(_text_response("Hello"))
        search = AsyncMock(return_value=_results())

        reply = await _run(gemini, {"web_search": search})

        assert reply == "Hello"
        tools = gemini.generate.await_args.kwargs["tools"]
        assert tools is not None
        declared = [d.name for tool in tools for d in tool.function_declarations]
        assert declared == ["web_search"]
        search.assert_not_awaited()


class TestToolExecution:
    """Tests for executing requested tools and feeding results back."""

    async def test_search_result_reaches_second_call(self) -> None:
        """Verify the tool runs and its results return to the model."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "alza akcie"})),
            _text_response("Answer"),
        )
        search = AsyncMock(return_value=_results("https://example.com/a"))

        reply = await _run(gemini, {"web_search": search})

        assert reply == "Answer"
        search.assert_awaited_once_with("alza akcie")
        contents = gemini.generate.await_args_list[1].args[0]
        function_call_part = contents[-2].parts[0]
        assert function_call_part.function_call.name == "web_search"
        function_response_part = contents[-1].parts[0]
        assert function_response_part.function_response.name == "web_search"
        assert "example.com/a" in str(function_response_part.function_response.response)

    async def test_parallel_calls_get_one_response_each(self) -> None:
        """Verify several calls in one turn all execute and report back."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q1"}), ("web_search", {"query": "q2"})),
            _text_response("Answer"),
        )
        search = AsyncMock(return_value=_results())

        await _run(gemini, {"web_search": search})

        assert search.await_count == 2
        contents = gemini.generate.await_args_list[1].args[0]
        response_parts = contents[-1].parts
        assert len(response_parts) == 2

    async def test_tool_error_reported_to_model(self) -> None:
        """Verify a failing tool becomes an error payload, not a crash."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q"})),
            _text_response("Answer without web"),
        )
        search = AsyncMock(side_effect=ToolError("rate limited"))

        reply = await _run(gemini, {"web_search": search})

        assert reply == "Answer without web"
        contents = gemini.generate.await_args_list[1].args[0]
        payload = str(contents[-1].parts[0].function_response.response)
        assert "rate limited" in payload


class TestFetchWhitelist:
    """Tests for the fetch_page url whitelist."""

    async def test_searched_url_may_be_fetched(self) -> None:
        """Verify a url returned by search is allowed through."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q"})),
            _call_response(("fetch_page", {"url": "https://example.com/a"})),
            _text_response("Answer"),
        )
        search = AsyncMock(return_value=_results("https://example.com/a"))
        fetch = AsyncMock(return_value="page text")

        reply = await _run(gemini, {"web_search": search, "fetch_page": fetch})

        assert reply == "Answer"
        fetch.assert_awaited_once_with("https://example.com/a")

    async def test_foreign_url_is_refused_without_calling(self) -> None:
        """Verify a url the search never returned is refused, not fetched."""
        gemini = _gemini(
            _call_response(("fetch_page", {"url": "http://metadata.google.internal/"})),
            _text_response("Answer"),
        )
        fetch = AsyncMock(return_value="secret")

        reply = await _run(gemini, {"fetch_page": fetch})

        assert reply == "Answer"
        fetch.assert_not_awaited()
        contents = gemini.generate.await_args_list[1].args[0]
        payload = str(contents[-1].parts[0].function_response.response)
        assert "error" in payload


class TestLimits:
    """Tests for the iteration and deadline guards."""

    async def test_iteration_limit_forces_toolless_answer(self) -> None:
        """Verify a looping model is cut off and asked for a final answer."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q1"})),
            _call_response(("web_search", {"query": "q2"})),
            _text_response("Forced answer"),
        )
        search = AsyncMock(return_value=_results())

        reply = await _run(gemini, {"web_search": search}, max_iterations=2)

        assert reply == "Forced answer"
        assert gemini.generate.await_count == 3
        assert gemini.generate.await_args_list[2].kwargs["tools"] is None

    async def test_deadline_forces_toolless_answer(self) -> None:
        """Verify an expired time budget ends the loop after the current round."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q"})),
            _text_response("Forced answer"),
        )
        search = AsyncMock(return_value=_results())

        reply = await _run(gemini, {"web_search": search}, deadline_seconds=0)

        assert reply == "Forced answer"
        assert gemini.generate.await_count == 2
        assert gemini.generate.await_args_list[1].kwargs["tools"] is None

    async def test_empty_final_text_raises_llm_error(self) -> None:
        """Verify a textless final response surfaces as LLMError."""
        gemini = _gemini(_text_response(""))

        with pytest.raises(LLMError):
            await _run(gemini, {})


class TestBlankArguments:
    """Tests for refusing degenerate tool arguments."""

    async def test_empty_query_is_refused_without_calling(self) -> None:
        """Verify a blank search query never reaches Brave."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "   "})),
            _text_response("Answer"),
        )
        search = AsyncMock(return_value=_results())

        reply = await _run(gemini, {"web_search": search})

        assert reply == "Answer"
        search.assert_not_awaited()
        contents = gemini.generate.await_args_list[1].args[0]
        assert "error" in str(contents[-1].parts[0].function_response.response)

    async def test_blank_result_url_never_enters_the_whitelist(self) -> None:
        """Verify an empty url from search cannot authorize an empty fetch."""
        gemini = _gemini(
            _call_response(("web_search", {"query": "q"})),
            _call_response(("fetch_page", {"url": ""})),
            _text_response("Answer"),
        )
        search = AsyncMock(return_value=_results(""))
        fetch = AsyncMock(return_value="text")

        await _run(gemini, {"web_search": search, "fetch_page": fetch})

        fetch.assert_not_awaited()
