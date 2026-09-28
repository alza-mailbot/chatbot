"""Bounded agent loop: the model may call web tools before answering."""

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from google.genai import types

from chatbot.core.attachments import Attachment
from chatbot.core.llm.gemini import GeminiClient, LLMError, build_contents
from chatbot.core.tools.web_search import SearchResult, ToolError
from chatbot.models.chat import ThreadMessage
from chatbot.utils.logger import logger

Tool = Callable[..., Awaitable[Any]]

_DECLARATIONS = {
    "web_search": types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current information.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
        },
    ),
    "fetch_page": types.FunctionDeclaration(
        name="fetch_page",
        description=(
            "Read the visible text of a page found by web_search. "
            "Only urls returned by web_search can be read."
        ),
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Url from web_search."}},
            "required": ["url"],
        },
    ),
}


async def run_agent(
    gemini: GeminiClient,
    *,
    subject: str,
    body: str,
    attachments: Sequence[Attachment] = (),
    thread: Sequence[ThreadMessage] = (),
    tools: Mapping[str, Tool],
    max_iterations: int,
    deadline_seconds: float,
) -> str:
    """Generate a reply, letting the model use the offered tools.

    Args:
        gemini: Client used for every model call.
        subject: Email subject line.
        body: Plain-text email body.
        attachments: Validated attachments included in the prompt.
        thread: Prior thread messages in chronological order.
        tools: Executable tools by declared name; empty means a single
            plain model call, identical to the pre-agent behaviour.
        max_iterations: Tool-call rounds before an answer is forced.
        deadline_seconds: Wall-clock budget for the whole loop.

    Returns:
        str: The final reply text.

    Raises:
        LLMError: If a model call fails or the final response has no text.
    """
    declarations = (
        [types.Tool(function_declarations=[_DECLARATIONS[n] for n in _DECLARATIONS if n in tools])]
        if tools
        else None
    )
    contents = build_contents(subject=subject, body=body, attachments=attachments, thread=thread)
    allowed_urls: set[str] = set()
    started = time.monotonic()

    for iteration in range(max_iterations):
        response = await gemini.generate(contents, tools=declarations)
        if not response.function_calls:
            return _final_text(response)
        if response.candidates and response.candidates[0].content:
            contents.append(response.candidates[0].content)
        parts = [
            await _execute(call, tools, allowed_urls, iteration) for call in response.function_calls
        ]
        contents.append(types.Content(role="user", parts=parts))
        if time.monotonic() - started >= deadline_seconds:
            logger.warning("[AGENT] deadline reached after iteration %d", iteration + 1)
            break

    logger.info("[AGENT] forcing a final answer without tools")
    return _final_text(await gemini.generate(contents, tools=None))


async def _execute(
    call: types.FunctionCall,
    tools: Mapping[str, Tool],
    allowed_urls: set[str],
    iteration: int,
) -> types.Part:
    """Run one requested tool call and wrap its outcome for the model."""
    name = str(call.name)
    args = dict(call.args or {})
    logger.info("[AGENT] iteration %d: tool %s(%r)", iteration + 1, name, args)
    payload = await _dispatch(name, args, tools, allowed_urls)
    return types.Part.from_function_response(name=name, response=payload)


async def _dispatch(
    name: str,
    args: dict[str, Any],
    tools: Mapping[str, Tool],
    allowed_urls: set[str],
) -> dict[str, Any]:
    """Execute the named tool and normalize its result or failure."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"Unknown tool {name!r}"}
    try:
        if name == "web_search":
            results: list[SearchResult] = await tool(str(args.get("query", "")))
            allowed_urls.update(result.url for result in results)
            return {"results": [result.model_dump() for result in results]}
        if name == "fetch_page":
            url = str(args.get("url", ""))
            # the whitelist keeps the model on pages web_search actually
            # returned: no server-side request forgery, no invented urls
            if url not in allowed_urls:
                return {"error": f"Url {url!r} was not returned by web_search"}
            return {"content": await tool(url)}
        return {"error": f"Tool {name!r} has no executor"}
    except ToolError as exc:
        logger.warning("[AGENT] tool %s failed: %s", name, exc)
        return {"error": str(exc)}


def _final_text(response: Any) -> str:
    """Return the response text or fail loudly when there is none."""
    if not response.text:
        raise LLMError("Gemini returned an empty response")
    return response.text
