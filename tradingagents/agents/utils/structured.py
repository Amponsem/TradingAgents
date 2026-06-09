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
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

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


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
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
    structured_llm: Optional[Any],
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
