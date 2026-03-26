"""Application service for processing chat turns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .config import settings
from .logging_utils import get_logger
from .rag_chain import build_rag_chain, format_sources
from .utils import sanitize_user_input
from .web_fallback import answer_with_google_fallback

logger = get_logger(__name__)


class SupportsInvoke(Protocol):
    """Minimal protocol for a LangChain-style runnable."""

    def invoke(self, values: dict[str, object]) -> dict[str, object]:
        """Execute the runnable."""


@dataclass(frozen=True)
class ChatTurnResult:
    """Normalized result of processing a chat turn."""

    user_input: str
    assistant_message: str
    next_chat_history: list[BaseMessage]
    used_fallback: bool = False


def _is_index_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "vector index not found",
        "vector index metadata is missing",
        "vector index was built with a different embedding setup",
        "run `python -m ottawa_assistant.retriever.ingest --use-seed` first",
    )
    return any(marker in message for marker in markers)


def _build_assistant_message(answer: str, sources: str, used_fallback: bool) -> str:
    source_block = sources or "- No official source available for this answer."
    if used_fallback:
        return (
            "_Local index unavailable. Used trusted Google search fallback._\n\n"
            f"{answer}\n\n**Sources**\n{source_block}"
        )
    return f"{answer}\n\n**Sources**\n{source_block}"


def process_chat_turn(
    user_input: str,
    chat_history: list[BaseMessage],
    *,
    rag_chain_factory: Callable[[], SupportsInvoke] = build_rag_chain,
    fallback_answerer: Callable[[str, list[BaseMessage]], tuple[str, str]] = answer_with_google_fallback,
) -> ChatTurnResult:
    """Validate input, run retrieval, and return the next assistant message."""
    cleaned_input = sanitize_user_input(user_input)
    if not cleaned_input:
        raise ValueError("Please enter a message before sending.")

    history_snapshot = list(chat_history)
    logger.info(
        "chat_turn_started question_length=%s history_messages=%s",
        len(cleaned_input),
        len(history_snapshot),
    )

    try:
        rag_chain = rag_chain_factory()
        result = rag_chain.invoke({"input": cleaned_input, "chat_history": history_snapshot})
        answer = str(result.get("answer", "")).strip() or "I could not find a confident answer."
        sources = format_sources(result.get("context", []))
        logger.info("chat_turn_completed mode=rag")
        return ChatTurnResult(
            user_input=cleaned_input,
            assistant_message=_build_assistant_message(answer, sources, used_fallback=False),
            next_chat_history=[
                *history_snapshot,
                HumanMessage(content=cleaned_input),
                AIMessage(content=answer),
            ],
            used_fallback=False,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        if settings.enable_web_fallback and _is_index_unavailable_error(exc):
            logger.warning("rag_chain_unavailable error=%s", exc)
            try:
                answer, sources = fallback_answerer(cleaned_input, history_snapshot)
                logger.info("chat_turn_completed mode=google_fallback")
                return ChatTurnResult(
                    user_input=cleaned_input,
                    assistant_message=_build_assistant_message(answer, sources, used_fallback=True),
                    next_chat_history=[
                        *history_snapshot,
                        HumanMessage(content=cleaned_input),
                        AIMessage(content=answer),
                    ],
                    used_fallback=True,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                logger.exception("google_fallback_failed error=%s", fallback_exc)
                return ChatTurnResult(
                    user_input=cleaned_input,
                    assistant_message=(
                        "Local index is unavailable and Google fallback also failed.\n\n"
                        f"Index error: `{exc}`\n\n"
                        f"Fallback error: `{fallback_exc}`\n\n"
                        "Try rebuilding local index:\n"
                        "`python -m ottawa_assistant.retriever.ingest --use-seed`"
                    ),
                    next_chat_history=history_snapshot,
                    used_fallback=False,
                )

        logger.exception("chat_turn_failed error=%s", exc)
        return ChatTurnResult(
            user_input=cleaned_input,
            assistant_message=f"I ran into an issue while preparing your answer.\n\nError: `{exc}`",
            next_chat_history=history_snapshot,
            used_fallback=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_turn_failed error=%s", exc)
        return ChatTurnResult(
            user_input=cleaned_input,
            assistant_message=f"I ran into an issue while preparing your answer.\n\nError: `{exc}`",
            next_chat_history=history_snapshot,
            used_fallback=False,
        )
