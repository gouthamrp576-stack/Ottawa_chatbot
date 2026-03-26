"""Workflow-oriented integration tests for the Ottawa assistant package."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

import ottawa_assistant.chat_service as chat_service
import ottawa_assistant.retriever.ingest as ingest_module


class _FakeRagChain:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result

    def invoke(self, values: dict[str, object]) -> dict[str, object]:
        assert "input" in values
        assert "chat_history" in values
        return self._result


def test_chat_service_rag_flow_appends_history_and_sources() -> None:
    existing_history = [HumanMessage(content="Previous question")]
    result = chat_service.process_chat_turn(
        "How do I find settlement services?",
        existing_history,
        rag_chain_factory=lambda: _FakeRagChain(
            {
                "answer": "Contact Ottawa newcomer settlement services.",
                "context": [
                    Document(
                        page_content="Settlement service information",
                        metadata={"source": "https://ottawa.ca/settlement"},
                    )
                ],
            }
        ),
    )

    assert "Contact Ottawa newcomer settlement services." in result.assistant_message
    assert "https://ottawa.ca/settlement" in result.assistant_message
    assert len(result.next_chat_history) == 3


def test_chat_service_fallback_flow_returns_fallback_banner() -> None:
    fallback_settings = replace(chat_service.settings, enable_web_fallback=True)
    original_settings = chat_service.settings
    chat_service.settings = fallback_settings
    try:
        result = chat_service.process_chat_turn(
            "Where can I get help with transit?",
            [],
            rag_chain_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("Vector index metadata is missing")
            ),
            fallback_answerer=lambda question, history: (
                "Use OC Transpo and newcomer support resources.",
                "- https://www.octranspo.com/",
            ),
        )
    finally:
        chat_service.settings = original_settings

    assert result.used_fallback is True
    assert "Used trusted Google search fallback" in result.assistant_message
    assert "https://www.octranspo.com/" in result.assistant_message


def test_ingest_main_expands_seed_urls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ingest_module,
        "parse_args",
        lambda: Namespace(use_seed=True, urls=["https://ottawa.ca/custom"], pdfs=["guide.pdf"]),
    )
    monkeypatch.setattr(
        ingest_module,
        "settings",
        replace(
            ingest_module.settings,
            seed_web_sources=("https://ottawa.ca/seed-one", "https://ontario.ca/seed-two"),
        ),
    )

    def fake_ingest(urls: list[str], pdfs: list[str]) -> None:
        captured["urls"] = urls
        captured["pdfs"] = pdfs

    monkeypatch.setattr(ingest_module, "ingest", fake_ingest)

    exit_code = ingest_module.main()

    assert exit_code == 0
    assert captured["urls"] == [
        "https://ottawa.ca/custom",
        "https://ottawa.ca/seed-one",
        "https://ontario.ca/seed-two",
    ]
    assert captured["pdfs"] == ["guide.pdf"]
