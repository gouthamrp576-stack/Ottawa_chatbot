"""Streamlit entrypoint for Ottawa Newcomer Assistant.

Run:
    PYTHONPATH=src streamlit run src/ottawa_assistant/main.py
"""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ottawa_assistant.config import settings, validate_settings
from ottawa_assistant.model_factory import runtime_summary
from ottawa_assistant.rag_chain import build_rag_chain, format_sources
from ottawa_assistant.web_fallback import answer_with_google_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSISTANT_MESSAGE = (
    "Welcome to Ottawa. Ask me anything about housing, healthcare, "
    "transportation, or newcomer services."
)


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
    """Render top hero section."""
    provider_text = (
        f"LLM: {settings.model_provider.upper()} | "
        f"Embeddings: {settings.embedding_provider.upper()}"
    )
    st.markdown(
        f"""
        <div class="canada-hero">
          <div class="canada-hero-mark">🍁</div>
          <div>
            <h1>Ottawa Newcomer Assistant</h1>
            <p>Friendly support for housing, healthcare, community services, and essential admin tasks.</p>
            <p class="canada-runtime">{provider_text}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    """Show quick official links in sidebar."""
    st.sidebar.markdown("## Runtime")
    st.sidebar.caption(runtime_summary())
    st.sidebar.divider()

    if st.sidebar.button("Reset conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}]
        st.session_state.chat_history = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("## Retrieval Mode")
    if settings.enable_web_fallback:
        st.sidebar.caption("Auto fallback to trusted Google search when local index is unavailable.")
    else:
        st.sidebar.caption("Local FAISS index only. No web fallback.")
    st.sidebar.divider()

    st.sidebar.markdown("## Quick official links")
    st.sidebar.markdown(
        """
        **Housing**
        - [Ontario tenant rights](https://www.ontario.ca/page/renting-ontario-your-rights)
        - [Ottawa settlement services](https://ottawa.ca/en/family-and-social-services/immigration-and-settlement)

        **Healthcare**
        - [Apply for OHIP](https://www.ontario.ca/page/apply-ohip-and-get-health-card)
        - [Ottawa Public Health](https://www.ottawapublichealth.ca/en/public-health-services.aspx)

        **Transportation**
        - [OC Transpo fares and payment](https://www.octranspo.com/en/fares/payment/where-how-to-pay/)
        - [City of Ottawa transit](https://ottawa.ca/en/parking-roads-and-travel/public-transit)

        **Newcomer support**
        - [OCISO](https://ociso.org/)
        - [YMCA-YWCA newcomer services](https://www.ymcaywca.ca/newcomer-services/)
        """
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


def _render_messages() -> None:
    """Render message history in chat area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _is_index_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "vector index not found",
        "vector index metadata is missing",
        "vector index was built with a different embedding setup",
        "run `python -m retriever.ingest --use-seed` first",
    )
    return any(marker in message for marker in markers)


def _process_input(user_input: str) -> None:
    """Invoke RAG chain and append assistant response to state."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking official Ottawa sources..."):
            answer = ""
            sources = ""
            used_fallback = False
            should_update_history = False

            try:
                rag_chain = _get_rag_chain()
                chat_history: list[BaseMessage] = st.session_state.chat_history
                # Invoke retrieval + answer generation with current conversation context.
                result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
                answer = str(result.get("answer", "")).strip() or "I could not find a confident answer."
                # Collect source links from retrieved documents.
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
                            "`python -m retriever.ingest --use-seed`"
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
                # Persist conversation history for the next retrieval turn.
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=answer))
                st.session_state.chat_history = chat_history

            st.markdown(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Ottawa Newcomer Assistant",
        page_icon="🍁",
        layout="centered",
    )
    _load_css()

    try:
        validate_settings(require_embeddings=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error: {exc}")
        st.info(
            "Check `.env` provider settings, then restart Streamlit.\n\n"
            "Tip: If you changed embedding provider/model, rebuild the index with:\n"
            "`python -m retriever.ingest --use-seed`"
        )
        return

    _render_header()
    _render_sidebar()
    _init_state()
    _render_messages()

    prompt = st.chat_input("Ask your question about settling in Ottawa...")
    if prompt:
        _process_input(prompt)


if __name__ == "__main__":
    main()
