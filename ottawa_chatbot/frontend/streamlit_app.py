import json
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_sources"
CHAT_STORE_PATH = PROJECT_ROOT / "frontend" / "chat_sessions.json"

API_BASE = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE}/chat/chat"
FEEDBACK_ENDPOINT = f"{API_BASE}/feedback"

st.set_page_config(
    page_title="Ottawa Newcomer Support Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 180px;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }

    /* Full app background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #eef6ff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #e5e7eb !important;
    }

    /* Sidebar selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [role="listbox"] {
        background: #ffffff !important;
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] [role="option"] {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] [role="option"]:hover {
        background: #e2e8f0 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: #1f2933 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #334155 !important;
        color: #ffffff !important;
    }

    /* Selected / normal chat buttons */
    .selected-chat button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28) !important;
        font-weight: 700 !important;
    }

    .normal-chat button {
        background: #1f2937 !important;
        color: #e5e7eb !important;
        border: 1px solid #374151 !important;
    }

    .normal-chat button:hover {
        background: #374151 !important;
        color: #ffffff !important;
    }

    /* FIXED HERO HEADER */
    .sticky-hero-wrap {
        position: fixed;
        top: 0;
        left: 21rem;   /* sidebar width */
        right: 0;
        z-index: 999;
        padding: 14px 32px 10px 32px;
        background: #eef6ff;
        border-bottom: 1px solid #dbeafe;
    }

    .hero-card {
        padding: 24px 28px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #172554 40%, #1e293b 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
        text-align: center;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 15px;
        line-height: 1.7;
        max-width: 900px;
        margin: 0 auto;
        opacity: 0.94;
    }

    /* Cards below header */
    .glass-card {
        margin-top: 16px;
        padding: 18px;
        border-radius: 22px;
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226,232,240,0.9);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .section-title {
        font-size: 16px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 12px;
    }

    .mini-note {
        font-size: 12px;
        color: #64748b;
    }

    .source-grid-title {
        font-size: 17px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .source-card {
        padding: 14px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
        transition: all 0.18s ease;
        margin-bottom: 12px;
    }

    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(15,23,42,0.10);
        border-color: #cbd5e1;
    }

    .source-title {
        font-size: 14px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 7px;
    }

    .source-line {
        font-size: 12px;
        color: #475569;
        margin-bottom: 4px;
        line-height: 1.55;
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 800;
        background: #eff6ff;
        color: #1d4ed8;
        margin-top: 6px;
    }

    .sidebar-title {
        font-size: 14px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    .chat-caption {
        font-size: 11px;
        color: #cbd5e1;
        margin-top: -2px;
        margin-bottom: 10px;
    }

    .chat-hint {
        font-size: 11px;
        color: #94a3b8;
    }

    .stButton > button {
        border-radius: 14px !important;
        font-weight: 650 !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        transition: all 0.18s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(15,23,42,0.10) !important;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
    }

    textarea, input, [data-baseweb="select"] {
        border-radius: 14px !important;
    }

    .empty-state {
        text-align: center;
        padding: 42px 16px 30px 16px;
        color: #475569;
    }

    .empty-title {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 10px;
    }

    .empty-sub {
        font-size: 14px;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def ensure_store_exists() -> None:
    CHAT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_STORE_PATH.exists():
        CHAT_STORE_PATH.write_text("[]", encoding="utf-8")


def load_chat_store() -> List[Dict[str, Any]]:
    ensure_store_exists()
    try:
        return json.loads(CHAT_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_chat_store(chats: List[Dict[str, Any]]) -> None:
    ensure_store_exists()
    CHAT_STORE_PATH.write_text(json.dumps(chats, indent=2, ensure_ascii=False), encoding="utf-8")


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def friendly_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        return dt.strftime("%d %b %Y • %H:%M")
    except Exception:
        return ts


def get_categories() -> List[str]:
    if not DATA_DIR.exists():
        return []
    cats = [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
    return sorted(cats)


def humanize_category(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def quick_questions_for_categories(categories: List[str]) -> List[str]:
    prompts = []
    for cat in categories:
        readable = humanize_category(cat).lower()
        prompts.append(f"What newcomer support information is available for {readable} in Ottawa?")
        prompts.append(f"Give me a summary of {readable} resources for newcomers in Ottawa.")
    # unique while preserving order
    seen = set()
    final = []
    for q in prompts:
        if q not in seen:
            seen.add(q)
            final.append(q)
    return final[:8]


def category_tint(value: str) -> str:
    palette = [
        "#dbeafe", "#dcfce7", "#fef3c7", "#ede9fe", "#fee2e2", "#cffafe", "#e2e8f0", "#fae8ff"
    ]
    idx = int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16) % len(palette)
    return palette[idx]


def category_text_color(value: str) -> str:
    palette = [
        "#1d4ed8", "#166534", "#92400e", "#6d28d9", "#b91c1c", "#0f766e", "#334155", "#a21caf"
    ]
    idx = int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16) % len(palette)
    return palette[idx]


def make_chat_title(first_message: str) -> str:
    title = first_message.strip().replace("\n", " ")
    if len(title) > 42:
        title = title[:42].rstrip() + "..."
    return title or "New Chat"


def new_chat_object() -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "selected_category": "All",
        "messages": [],
    }


def get_active_chat(chats: List[Dict[str, Any]], chat_id: str) -> Dict[str, Any]:
    for chat in chats:
        if chat["id"] == chat_id:
            return chat
    if chats:
        return chats[0]
    fresh = new_chat_object()
    chats.append(fresh)
    return fresh


def persist_session() -> None:
    save_chat_store(st.session_state.chat_sessions)


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def create_new_chat() -> None:
    fresh = new_chat_object()
    st.session_state.chat_sessions.insert(0, fresh)
    st.session_state.active_chat_id = fresh["id"]
    persist_session()


def delete_chat(chat_id: str) -> None:
    sessions = [c for c in st.session_state.chat_sessions if c["id"] != chat_id]
    st.session_state.chat_sessions = sessions if sessions else [new_chat_object()]
    st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]
    persist_session()


def build_prompt(user_input: str, category: str) -> str:
    if category and category != "All":
        return f"[Category: {category}] {user_input}"
    return user_input


def call_chat_api(message: str, category: str | None = None) -> Dict[str, Any]:
    payload = {
        "message": message,
        "category": category if category and category != "All" else None,
    }
    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def send_feedback(answer: str, message: str, rating: str) -> None:
    payload = {
        "message": message,
        "answer": answer,
        "rating": rating,
        "timestamp": now_iso(),
    }
    try:
        requests.post(FEEDBACK_ENDPOINT, json=payload, timeout=6)
    except Exception:
        pass


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    st.markdown('<div class="source-grid-title">Sources</div>', unsafe_allow_html=True)

    for src in sources:
        file_name = src.get("file_name", "Unknown File")
        title = src.get("title", "") or file_name
        url = src.get("url", "")
        category = src.get("category", "") or "Unknown"

        tint = category_tint(category)
        txt = category_text_color(category)

        if url:
            link_html = f'<a href="{url}" target="_blank">Open official source</a>'
        else:
            link_html = "Official URL will appear after metadata integration"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{title}</div>
                <div class="source-line"><strong>File:</strong> {file_name}</div>
                <div class="source-line"><strong>Source:</strong> {link_html}</div>
                <div class="source-line">
                    <span class="badge" style="background:{tint}; color:{txt};">{humanize_category(category)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# STATE INIT
# =========================================================
if "chat_sessions" not in st.session_state:
    loaded = load_chat_store()
    st.session_state.chat_sessions = loaded if loaded else [new_chat_object()]

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]

categories = get_categories()
category_options = ["All"] + categories

active_chat = get_active_chat(st.session_state.chat_sessions, st.session_state.active_chat_id)

# normalize selected category
if active_chat.get("selected_category", "All") not in category_options:
    active_chat["selected_category"] = "All"

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Chats")

    if st.button("➕  New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown('<div class="chat-caption">Recent chats saved locally in this project.</div>', unsafe_allow_html=True)

    # recent chats
    for idx, chat in enumerate(st.session_state.chat_sessions):
        is_active = chat["id"] == st.session_state.active_chat_id
        title = chat.get("title", "New Chat")
        updated = friendly_time(chat.get("updated_at", ""))

        row_cols = st.columns([6, 1], gap="small")

        with row_cols[0]:
            label = f"🟦 {title}" if is_active else f"💬 {title}"

            wrapper_class = "selected-chat" if is_active else "normal-chat"
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)

            if st.button(label, key=f"open_chat_{chat['id']}", use_container_width=True):
                set_active_chat(chat["id"])
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        with row_cols[1]:
            if st.button("✕", key=f"delete_chat_{chat['id']}"):
                delete_chat(chat["id"])
                st.rerun()

        st.caption(updated)

    st.markdown("---")
    st.markdown("### Categories")

    selected_index = category_options.index(active_chat.get("selected_category", "All"))
    chosen = st.selectbox(
        "Filter context",
        options=category_options,
        index=selected_index,
        format_func=lambda x: "All Topics" if x == "All" else humanize_category(x),
        label_visibility="collapsed",
        key="sidebar_category_select",
    )

    if chosen != active_chat.get("selected_category", "All"):
        active_chat["selected_category"] = chosen
        active_chat["updated_at"] = now_iso()
        persist_session()

    st.markdown("---")
    st.markdown("### Quick Questions")

    quick_questions = quick_questions_for_categories(categories)
    for q in quick_questions:
        if st.button(q, key=f"qq_{q}", use_container_width=True):
            st.session_state.prefill_prompt = q

# =========================================================
# MAIN UI
# =========================================================

st.markdown(
    """
    <div class="sticky-hero-wrap">
        <div class="hero-card">
            <div class="hero-title">Ottawa Newcomer Support Chatbot</div>
            <div class="hero-subtitle">
                A trusted AI assistant that helps newcomers in Ottawa explore healthcare, housing,
                jobs, study, transportation, community, and settlement support using verified document sources.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([3, 2], gap="large")

with top_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Current Conversation</div>', unsafe_allow_html=True)
    st.write(f"**Title:** {active_chat.get('title', 'New Chat')}")
    st.write(
        f"**Topic Filter:** {'All Topics' if active_chat.get('selected_category', 'All') == 'All' else humanize_category(active_chat.get('selected_category', 'All'))}"
    )
    st.markdown(
        f"<div class='mini-note'>Created: {friendly_time(active_chat.get('created_at', ''))} &nbsp;&nbsp;•&nbsp;&nbsp; Updated: {friendly_time(active_chat.get('updated_at', ''))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)

    with qa1:
        if st.button("Clear Current Chat", use_container_width=True):
            active_chat["messages"] = []
            active_chat["title"] = "New Chat"
            active_chat["updated_at"] = now_iso()
            persist_session()
            st.rerun()

    with qa2:
        if st.button("Duplicate Chat", use_container_width=True):
            copy_chat = json.loads(json.dumps(active_chat))
            copy_chat["id"] = str(uuid.uuid4())
            copy_chat["title"] = active_chat.get("title", "New Chat") + " Copy"
            copy_chat["created_at"] = now_iso()
            copy_chat["updated_at"] = now_iso()
            st.session_state.chat_sessions.insert(0, copy_chat)
            st.session_state.active_chat_id = copy_chat["id"]
            persist_session()
            st.rerun()

    with qa3:
        last_user = ""
        for msg in reversed(active_chat.get("messages", [])):
            if msg["role"] == "user":
                last_user = msg["content"]
                break
        if st.button("Retry Last", use_container_width=True, disabled=not bool(last_user)):
            st.session_state.prefill_prompt = last_user
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CHAT HISTORY
# =========================================================
messages = active_chat.get("messages", [])

if not messages:
    st.markdown(
        """
        <div class="empty-state">
            
        </div>
        """,
        unsafe_allow_html=True,
    )

for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

            fb1, fb2, _ = st.columns([1, 1, 8])
            with fb1:
                if st.button("👍 Helpful", key=f"up_{active_chat['id']}_{idx}"):
                    prev_user = ""
                    for j in range(idx - 1, -1, -1):
                        if messages[j]["role"] == "user":
                            prev_user = messages[j]["content"]
                            break
                    send_feedback(msg["content"], prev_user, "up")
                    st.success("Feedback sent")

            with fb2:
                if st.button("👎 Needs Work", key=f"down_{active_chat['id']}_{idx}"):
                    prev_user = ""
                    for j in range(idx - 1, -1, -1):
                        if messages[j]["role"] == "user":
                            prev_user = messages[j]["content"]
                            break
                    send_feedback(msg["content"], prev_user, "down")
                    st.warning("Feedback sent")

# =========================================================
# INPUT
# =========================================================
prefill = st.session_state.pop("prefill_prompt", None)
user_input = st.chat_input("Ask a question about newcomer support in Ottawa...")

if prefill and not user_input:
    user_input = prefill

if user_input:
    if active_chat["title"] == "New Chat":
        active_chat["title"] = make_chat_title(user_input)

    final_prompt = build_prompt(user_input, active_chat.get("selected_category", "All"))

    user_msg = {
        "role": "user",
        "content": user_input,
        "timestamp": now_iso(),
    }
    active_chat["messages"].append(user_msg)
    active_chat["updated_at"] = now_iso()
    persist_session()

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching trusted documents and generating response..."):
            try:
                selected_category = active_chat.get("selected_category", "All")
                result = call_chat_api(user_input, selected_category)
                answer = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.markdown(answer)
                render_sources(sources)

                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "timestamp": now_iso(),
                }
                active_chat["messages"].append(assistant_msg)
                active_chat["updated_at"] = now_iso()
                persist_session()

            except Exception:
                error_msg = (
                    "The chatbot service is currently unavailable. "
                    "Please try again in a moment."
                )
                st.error(error_msg)

                assistant_msg = {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "timestamp": now_iso(),
                }
                active_chat["messages"].append(assistant_msg)
                active_chat["updated_at"] = now_iso()
                persist_session()

st.markdown('</div>', unsafe_allow_html=True)