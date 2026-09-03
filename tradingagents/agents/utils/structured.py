"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# How many extra times to re-attempt a tool-bound analyst call before dropping tools.
# The tool-format failures on weak local models are stochastic, so a re-sample usually
# succeeds and preserves real data access. Env-overridable; 2 retries = up to 3 tries.
_TOOL_CALL_RETRIES = int(os.environ.get("TA_TOOL_CALL_RETRIES", "2"))

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)

# Models that have already failed a structured-output call. Local models served via
# LM Studio / Ollama reliably ignore the JSON schema and return markdown, so the
# structured call fails *every* time and we waste a full (often slow) generation on
# it before falling back. Once a given model fails, we skip the structured attempt
# for it from then on and go straight to free-text — the downstream parser
# (runner._parse_pm_text) extracts the rating from the markdown either way. Keyed on
# the model name so it persists across agents and runs within the process.
_structured_disabled_models: set[str] = set()


def _model_key(llm: Any) -> Optional[str]:
    for attr in ("model_name", "model", "model_id"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    model_key = _model_key(plain_llm) or _model_key(structured_llm)
    attempt_structured = structured_llm is not None and (
        model_key is None or model_key not in _structured_disabled_models
    )
    if attempt_structured:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            return render(result)
        except Exception as exc:
            if model_key:
                _structured_disabled_models.add(model_key)
            logger.warning(
                "%s: structured-output invocation failed for model '%s' (%s); using "
                "free text for this model from now on (skips the wasted retry)",
                agent_name, model_key or "?", exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content


def invoke_analyst_with_tools(
    prompt: Any,
    llm: Any,
    tools: list,
    messages: Any,
    agent_name: str,
) -> Any:
    """Invoke an analyst's tool-bound chain resiliently and return the AIMessage.

    Data-fetching analysts (market/news/fundamentals) bind their tools and let the
    model decide which to call. Weak local models served via LM Studio (e.g. gemma
    finetunes) intermittently emit a tool call the server cannot parse — LM Studio
    then 400s with "...does not match the expected peg-gemma4 format". A 400 is not
    retried by the OpenAI SDK, so a single such miss at any analyst node aborts the
    entire multi-agent run (every ticker comes back "Hold (failed)").

    The failure is stochastic, not deterministic (the same model succeeds on most
    tickers), so we simply re-sample the tool-bound call a couple of times; a retry
    usually parses cleanly and preserves real data access. Only if every attempt
    fails do we drop the tools and take a plain free-text pass, so the graph still
    advances (with a thinner, data-light report) instead of the run dying outright.
    """
    tool_chain = prompt | llm.bind_tools(tools)
    last_exc: Optional[Exception] = None
    for attempt in range(_TOOL_CALL_RETRIES + 1):
        try:
            return tool_chain.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — keep the run alive; see docstring
            last_exc = exc
            logger.warning(
                "%s: tool-bound invocation failed (attempt %d/%d): %s",
                agent_name, attempt + 1, _TOOL_CALL_RETRIES + 1, exc,
            )
    logger.warning(
        "%s: dropping tools after %d failed attempts — free-text pass so the run "
        "continues (last error: %s)", agent_name, _TOOL_CALL_RETRIES + 1, last_exc,
    )
    return (prompt | llm).invoke(messages)
