"""Tests for ottawa_assistant.chat_service."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

import ottawa_assistant.chat_service as chat_service


class _FakeRagChain:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result

    def invoke(self, values: dict[str, object]) -> dict[str, object]:
        assert "input" in values
        assert "chat_history" in values
        return self._result


class TestProcessChatTurn:
    def test_returns_rag_answer_and_updates_history(self) -> None:
        result = chat_service.process_chat_turn(
            "How do I apply for OHIP?",
            [],
            rag_chain_factory=lambda: _FakeRagChain(
                {
                    "answer": "Apply through Ontario's health card process.",
                    "context": [
                        Document(
                            page_content="OHIP info",
                            metadata={"source": "https://ontario.ca/ohip"},
                        )
                    ],
                }
            ),
        )

        assert result.user_input == "How do I apply for OHIP?"
        assert "Apply through Ontario's health card process." in result.assistant_message
        assert "https://ontario.ca/ohip" in result.assistant_message
        assert len(result.next_chat_history) == 2
        assert result.used_fallback is False

    def test_uses_google_fallback_when_index_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            chat_service,
            "settings",
            replace(chat_service.settings, enable_web_fallback=True),
        )

        result = chat_service.process_chat_turn(
            "Find newcomer services",
            [],
            rag_chain_factory=lambda: (_ for _ in ()).throw(
                FileNotFoundError("Vector index not found at /tmp/index")
            ),
            fallback_answerer=lambda question, history: (
                "Fallback answer",
                "- https://ottawa.ca/newcomers",
            ),
        )

        assert result.used_fallback is True
        assert "Fallback answer" in result.assistant_message
        assert "trusted Google search fallback" in result.assistant_message
        assert len(result.next_chat_history) == 2

    def test_returns_error_message_on_processing_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            chat_service,
            "settings",
            replace(chat_service.settings, enable_web_fallback=False),
        )

        result = chat_service.process_chat_turn(
            "Can you help me?",
            [],
            rag_chain_factory=lambda: (_ for _ in ()).throw(RuntimeError("upstream failure")),
        )

        assert "I ran into an issue while preparing your answer." in result.assistant_message
        assert result.next_chat_history == []

    def test_rejects_empty_input_after_sanitization(self) -> None:
        with pytest.raises(ValueError, match="Please enter a message"):
            chat_service.process_chat_turn(" \n\t ", [])

    def test_rejects_input_that_exceeds_length_limit(self) -> None:
        with pytest.raises(ValueError, match="maximum length"):
            chat_service.process_chat_turn("x" * 2001, [])
