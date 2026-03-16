"""Streamlit entrypoint for Ottawa Newcomer Assistant.

Run:
    PYTHONPATH=src streamlit run src/ottawa_assistant/main.py
"""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import Final

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import sys
print(sys.path)
from ottawa_assistant.config import settings, validate_settings
from ottawa_assistant.model_factory import runtime_summary
from ottawa_assistant.rag_chain import build_rag_chain, format_sources
from ottawa_assistant.web_fallback import answer_with_google_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSISTANT_MESSAGE = (
    "Hi! I'm OttawaBot. I can help newcomers with jobs, study options, "
    "healthcare, housing, and local services."
)
QUICK_PROMPTS: Final[list[tuple[str, str]]] = [
    ("Jobs in Ottawa", "Can you help me find a job in Ottawa?"),
    ("Student Resources", "Show me study resources for newcomers in Ottawa."),
    ("Tenant Rights", "What are my tenant rights in Ontario?"),
    ("Healthcare", "How do I apply for OHIP in Ottawa?"),
]


@st.cache_data(show_spinner=False)
def _build_css() -> str:
    """Read CSS once and inject maple leaf data URI."""
    css_path = PROJECT_ROOT / "public" / "style.css"
    leaf_svg_path = PROJECT_ROOT / "public" / "maple-leaf.svg"
    if not css_path.exists() or not leaf_svg_path.exists():
        return ""

    css_text = css_path.read_text(encoding="utf-8")
    leaf_b64 = b64encode(leaf_svg_path.read_bytes()).decode("utf-8")
    css_text = css_text.replace(
        "__MAPLE_LEAF_DATA_URI__",
        f"data:image/svg+xml;base64,{leaf_b64}",
    )
    return css_text


def _load_css() -> None:
    """Inject Streamlit CSS into page."""
    css_text = _build_css()
    if not css_text:
        return
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _get_rag_chain():
    """Create the LangChain retrieval pipeline once per Streamlit server process."""
    return build_rag_chain()


def _render_header() -> None:
    """Render top sky banner section."""
    provider_text = (
        f"LLM: {settings.model_provider.upper()} | "
        f"Embeddings: {settings.embedding_provider.upper()}"
    )
    st.markdown(
        f"""
        <section class="ott-top-banner">
          <div class="ott-banner-skyline"></div>
          <div>
            <h1>Welcome to Ottawa!</h1>
            <p>OttawaBot helps newcomers settle with clear, trusted guidance.</p>
            <p class="ott-runtime">{provider_text}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_left_rail() -> None:
    """Render navigation-style left rail."""
    st.markdown(
        """
        <section class="ott-left-rail">
          <div class="ott-left-group">
            <div class="ott-left-item is-active">💼 Jobs</div>
            <div class="ott-left-item">🎓 Study Resources</div>
            <div class="ott-left-item">💚 Healthcare</div>
            <div class="ott-left-item">🏙️ Local Services</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Find a Job", use_container_width=True):
        _process_input("Can you help me find a job in Ottawa?")
        st.rerun()
    if st.button("Study Resources", use_container_width=True):
        _process_input("Show me newcomer-friendly study resources in Ottawa.")
        st.rerun()
    if st.button("Housing", use_container_width=True):
        _process_input("Can you help me understand renting and housing options in Ottawa?")
        st.rerun()
    if st.button("Local Services", use_container_width=True):
        _process_input("What local newcomer services should I contact first in Ottawa?")
        st.rerun()
    if st.button("Ask Anything", use_container_width=True):
        _process_input("What should I do first week after arriving in Ottawa as a newcomer?")
        st.rerun()

    st.markdown(
        """
        <div class="ott-left-footer">🍁 OttawaBot is typing...</div>
        """
        ,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    """Initialize chat state containers for UI and retriever history."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": DEFAULT_ASSISTANT_MESSAGE,
            }
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _render_chat_intro() -> None:
    """Render welcome row in the chat panel."""
    st.markdown(
        """
        <section class="ott-chat-intro">
          <div class="ott-bot-avatar">🤖</div>
          <div class="ott-chat-intro-bubble">
            <strong>Hi! I'm OttawaBot</strong> - here to help newcomers settle in Ottawa.
            How can I assist you today?
          </div>
          <div class="ott-user-tag">Minh Alice Nguyen</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_messages() -> None:
    """Render message history in chat panel."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_resource_cards() -> None:
    """Render static resource cards matching the visual concept."""
    st.markdown(
        """
        <section class="ott-resource-grid">
          <article class="ott-resource-card">
            <h4>Job Bank Listings</h4>
            <p>Find jobs on Job Bank Canada and filter by Ottawa, language, and experience level.</p>
            <a href="https://www.jobbank.gc.ca/" target="_blank">Visit Job Bank</a>
          </article>
          <article class="ott-resource-card">
            <h4>Employment Ontario</h4>
            <p>Career support, resume help, and training pathways for job seekers in Ontario.</p>
            <a href="https://www.ontario.ca/page/employment-ontario" target="_blank">Visit Employment Ontario</a>
          </article>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_quick_prompts() -> None:
    """Render quick suggestion chips."""
    cols = st.columns(4, gap="small")
    for col, (label, prompt) in zip(cols, QUICK_PROMPTS):
        with col:
            if st.button(label, use_container_width=True, key=f"quick-{label}"):
                _process_input(prompt)
                st.rerun()


def _render_input_form() -> None:
    """Render message form styled like inline chat input."""
    with st.form("ottawa-chat-form", clear_on_submit=True):
        user_input = st.text_input(
            label="Type your message",
            placeholder="Type a message...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        _process_input(user_input.strip())
        st.rerun()


def _is_index_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "vector index not found",
        "vector index metadata is missing",
        "vector index was built with a different embedding setup",
        "run `python -m ottawa_assistant.retriever.ingest --use-seed` first",
    )
    return any(marker in message for marker in markers)


def _process_input(user_input: str) -> None:
    """Invoke RAG chain and append assistant response to state."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Checking official Ottawa sources..."):
        answer = ""
        sources = ""
        used_fallback = False
        should_update_history = False

        try:
            rag_chain = _get_rag_chain()
            chat_history: list[BaseMessage] = st.session_state.chat_history
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            answer = str(result.get("answer", "")).strip() or "I could not find a confident answer."
            sources = format_sources(result.get("context", []))
            should_update_history = True
        except (FileNotFoundError, RuntimeError) as exc:
            chat_history = st.session_state.chat_history
            if settings.enable_web_fallback and _is_index_unavailable_error(exc):
                try:
                    answer, sources = answer_with_google_fallback(
                        question=user_input,
                        chat_history=chat_history,
                    )
                    used_fallback = True
                    should_update_history = True
                except Exception as fallback_exc:  # noqa: BLE001
                    final_text = (
                        "Local index is unavailable and Google fallback also failed.\n\n"
                        f"Index error: `{exc}`\n\n"
                        f"Fallback error: `{fallback_exc}`\n\n"
                        "Try rebuilding local index:\n"
                        "`python -m ottawa_assistant.retriever.ingest --use-seed`"
                    )
            else:
                final_text = f"I ran into an issue while preparing your answer.\n\nError: `{exc}`"
        except Exception as exc:  # noqa: BLE001
            final_text = f"I ran into an issue while preparing your answer.\n\nError: `{exc}`"

    if should_update_history:
        if used_fallback:
            final_text = (
                "_Local index unavailable. Used trusted Google search fallback._\n\n"
                f"{answer}\n\n**Sources**\n{sources}"
            )
        else:
            final_text = f"{answer}\n\n**Sources**\n{sources}"
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        st.session_state.chat_history = chat_history

    st.session_state.messages.append({"role": "assistant", "content": final_text})


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Ottawa Newcomer Assistant",
        page_icon="🍁",
        layout="wide",
    )
    _load_css()

    try:
        validate_settings(require_embeddings=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error: {exc}")
        st.info(
            "Check `.env` provider settings, then restart Streamlit.\n\n"
            "Tip: If you changed embedding provider/model, rebuild the index with:\n"
            "`python -m ottawa_assistant.retriever.ingest --use-seed`"
        )
        return

    _render_header()
    _init_state()

    left_col, right_col = st.columns([1, 3.2], gap="small")
    with left_col:
        _render_left_rail()
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]
            st.session_state.chat_history = []
            st.rerun()
        st.caption(runtime_summary())

    with right_col:
        _render_chat_intro()
        _render_messages()
        _render_resource_cards()
        _render_quick_prompts()
        _render_input_form()


if __name__ == "__main__":
    main()
