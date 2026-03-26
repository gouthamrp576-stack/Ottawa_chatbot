"""Tests for ottawa_assistant.prompts — prompt template construction."""

from ottawa_assistant.prompts import build_contextualize_prompt, build_qa_prompt


class TestBuildContextualizePrompt:

    def test_returns_prompt_template(self):
        prompt = build_contextualize_prompt()
        assert prompt is not None

    def test_includes_chat_history_placeholder(self):
        prompt = build_contextualize_prompt()
        input_vars = prompt.input_variables
        # MessagesPlaceholder adds chat_history
        assert "chat_history" in input_vars or any(
            hasattr(m, "variable_name") and m.variable_name == "chat_history"
            for m in prompt.messages
        )

    def test_includes_input_variable(self):
        prompt = build_contextualize_prompt()
        assert "input" in prompt.input_variables

    def test_system_message_mentions_rewrite(self):
        prompt = build_contextualize_prompt()
        system_content = prompt.messages[0].prompt.template
        assert "standalone" in system_content.lower() or "rewrite" in system_content.lower()


class TestBuildQaPrompt:

    def test_returns_prompt_template(self):
        prompt = build_qa_prompt()
        assert prompt is not None

    def test_includes_context_variable(self):
        prompt = build_qa_prompt()
        # context is in the system message template
        system_content = prompt.messages[0].prompt.template
        assert "{context}" in system_content

    def test_includes_input_variable(self):
        prompt = build_qa_prompt()
        assert "input" in prompt.input_variables

    def test_system_prompt_mentions_newcomer(self):
        prompt = build_qa_prompt()
        system_content = prompt.messages[0].prompt.template
        assert "newcomer" in system_content.lower()

    def test_system_prompt_has_grounding_rules(self):
        prompt = build_qa_prompt()
        system_content = prompt.messages[0].prompt.template.lower()
        assert "never invent" in system_content or "only the provided context" in system_content
