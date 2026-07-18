import os
import uuid
from datetime import date

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f1220 0%, #161a2e 45%, #1b2038 100%);
    }

    /* Hero banner */
    .app-hero {
        background: linear-gradient(120deg, #6a5cff 0%, #8a5cff 35%, #ff6bb3 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(106, 92, 255, 0.35);
    }
    .app-hero h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .app-hero p {
        color: rgba(255,255,255,0.9);
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161a2e 0%, #10132200 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
    }

    /* Cards / containers with border */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.03);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.05s ease-in-out;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-purple { background: rgba(138, 92, 255, 0.18); color: #c6b6ff; }
    .badge-pink   { background: rgba(255, 107, 179, 0.18); color: #ffb3d9; }
    .badge-green  { background: rgba(60, 210, 150, 0.18); color: #8be9c4; }
    .badge-blue   { background: rgba(80, 160, 255, 0.18); color: #a9d0ff; }

    /* Metric-style thread id chip */
    .thread-chip {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        font-family: 'SFMono-Regular', Consolas, monospace;
        font-size: 0.85rem;
        color: #d8d8ff;
    }

    /* Booking approval card (Human-in-the-Loop) */
    .booking-card {
        background: linear-gradient(135deg, #1c2140 0%, #241a3d 100%);
        border: 1px solid rgba(138, 92, 255, 0.35);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .booking-card-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: #a99bff;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .booking-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    .booking-route-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .booking-mode-icon {
        font-size: 2.1rem;
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 0.5rem 0.7rem;
    }
    .booking-route {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .booking-sub {
        color: #b7b3d9;
        font-size: 0.88rem;
        margin-top: 0.15rem;
    }
    .booking-price {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(90deg, #b78dff, #ff8bc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap;
    }
    .booking-question {
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        font-weight: 600;
        color: #eae7ff;
    }

    /* Primary (Yes) button accent */
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #3cd296, #29b8a0);
    }

    /* --- Kill leftover white/blue defaults on widgets --- */
    /* Text / date / number inputs */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.05) !important;
        color: #eae7ff !important;
        border-color: rgba(138, 92, 255, 0.35) !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stDateInput input:focus,
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #8a5cff !important;
        box-shadow: 0 0 0 1px #8a5cff !important;
    }

    /* Date picker popover calendar */
    div[data-baseweb="calendar"],
    div[data-baseweb="popover"] div[data-baseweb="calendar"] {
        background: #1b2038 !important;
        color: #eae7ff !important;
    }
    div[data-baseweb="calendar"] [aria-selected="true"] {
        background: #8a5cff !important;
    }

    /* Dataframe / table (Previous Bookings) */
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px;
    }

    /* Radio selected accent -> purple instead of default blue/red */
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] > div:first-child {
        background-color: #8a5cff !important;
        border-color: #8a5cff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_MODE_ICONS = {"flight": "✈️", "train": "🚆", "bus": "🚌"}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None
if "last_asked_field" not in st.session_state:
    st.session_state.last_asked_field = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Pranav"
if "user_email" not in st.session_state:
    st.session_state.user_email = "pranavkumarpk0107@gmail.com"

with st.sidebar:
    st.markdown("### ✈️ AI Travel Planner")
    st.caption("LangGraph-powered travel assistant")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["💬 Chat", "🧾 Previous Bookings", "🧠 Memory Demo"],
        label_visibility="collapsed",
    )

    with st.expander("👤 Passenger details"):
        st.session_state.user_name = st.text_input("Name", st.session_state.user_name)
        st.session_state.user_email = st.text_input("Email", st.session_state.user_email)

    st.markdown("---")
    st.caption("Session")
    st.markdown(f'<span class="thread-chip">🧵 {st.session_state.thread_id[:8]}</span>', unsafe_allow_html=True)
    if st.button("🔄 Start new conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.session_state.last_asked_field = None
        st.rerun()

page = page.split(" ", 1)[1] if " " in page else page


def call_chat(message: str):
    resp = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "thread_id": st.session_state.thread_id,
            "message": message,
            "user_name": st.session_state.user_name,
            "user_email": st.session_state.user_email,
        },
        timeout=120,
    )
    return _unwrap(resp)


def call_resume(approved: bool):
    resp = requests.post(
        f"{API_BASE_URL}/chat/resume",
        json={"thread_id": st.session_state.thread_id, "approved": approved},
        timeout=120,
    )
    return _unwrap(resp)


def _unwrap(resp: requests.Response) -> dict:
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except ValueError:
            body = {"error": resp.text}
        st.error(f"Backend error: {body.get('error', 'unknown error')}")
        with st.expander("Traceback"):
            st.code(body.get("traceback", "(no traceback returned)"))
        st.stop()
    return resp.json()


def handle_result(result: dict):
    if result["status"] == "interrupted":
        st.session_state.pending_interrupt = result["interrupt"]
        st.session_state.last_asked_field = None
    else:
        st.session_state.pending_interrupt = None
        st.session_state.last_asked_field = result.get("state", {}).get("last_asked_field")
        if result.get("reply"):
            st.session_state.chat_history.append(("assistant", result["reply"]))


if page == "Chat":
    st.markdown(
        """
        <div class="app-hero">
            <h1>💬 Chat with your Travel Planner</h1>
            <p>Try: "I want to travel from Bangalore to Goa" and answer the follow-up questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge badge-purple">Intent detection</span>'
        '<span class="badge badge-pink">Memory</span>'
        '<span class="badge badge-green">Tool calling</span>'
        '<span class="badge badge-blue">Human-in-the-loop</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    for role, text in st.session_state.chat_history:
        avatar = "🧑‍💼" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(text)

    if st.session_state.pending_interrupt:
        interrupt_data = st.session_state.pending_interrupt
        mode = str(interrupt_data.get("mode") or "").lower()
        icon = _MODE_ICONS.get(mode, "🧭")
        price = interrupt_data.get("price")
        price_str = f"₹{price:,.0f}" if isinstance(price, (int, float)) else "—"

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                f"""
                <div class="booking-card">
                    <div class="booking-card-label">🧾 Booking summary · awaiting your approval</div>
                    <div class="booking-card-top">
                        <div class="booking-route-row">
                            <div class="booking-mode-icon">{icon}</div>
                            <div>
                                <div class="booking-route">{interrupt_data.get('route', '—')}</div>
                                <div class="booking-sub">{mode.title() or '—'} &nbsp;·&nbsp; {interrupt_data.get('date', '—')}</div>
                            </div>
                        </div>
                        <div class="booking-price">{price_str}</div>
                    </div>
                    <div class="booking-question">{interrupt_data.get('question', 'Proceed with this booking?')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            if col1.button("✅ Yes, book it", use_container_width=True, type="primary"):
                with st.spinner("Booking..."):
                    result = call_resume(True)
                handle_result(result)
                st.rerun()
            if col2.button("❌ No, cancel", use_container_width=True):
                with st.spinner("Cancelling..."):
                    result = call_resume(False)
                handle_result(result)
                st.rerun()

    else:
        if st.session_state.last_asked_field == "travel_date":
            with st.container(border=True):
                st.caption("📅 Pick a date, or just type one in the chat box below.")
                dcol1, dcol2 = st.columns([3, 1])
                picked_date = dcol1.date_input(
                    "Travel date",
                    value=None,
                    min_value=date.today(),
                    label_visibility="collapsed",
                )
                if dcol2.button("Use this date", use_container_width=True) and picked_date:
                    date_str = picked_date.strftime("%Y-%m-%d")
                    st.session_state.chat_history.append(("user", date_str))
                    with st.spinner("Thinking..."):
                        result = call_chat(date_str)
                    handle_result(result)
                    st.rerun()

        user_input = st.chat_input("Where would you like to travel?")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.spinner("Thinking..."):
                result = call_chat(user_input)
            handle_result(result)
            st.rerun()

elif page == "Previous Bookings":
    st.markdown(
        """
        <div class="app-hero">
            <h1>🧾 Previous Bookings</h1>
            <p>Fetched directly from the database — no LLM involved on this page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh"):
        st.rerun()

    bookings = []
    try:
        resp = requests.get(f"{API_BASE_URL}/bookings/{st.session_state.user_email}", timeout=30)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text}
            st.error(f"Backend error: {body.get('error', 'unknown error')}")
            with st.expander("Traceback"):
                st.code(body.get("traceback", "(no traceback returned)"))
        else:
            bookings = resp.json()
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")

    if not bookings:
        with st.container(border=True):
            st.info("No bookings yet. Book a trip in the Chat tab first.")
    else:
        try:
            st.dataframe(bookings, use_container_width=True)
        except Exception as exc:
            st.warning(f"Couldn't render as a table ({exc}); showing raw data instead.")
            st.json(bookings)

elif page == "Memory Demo":
    try:
        st.markdown(
            """
            <div class="app-hero">
                <h1>🧠 Memory & Checkpointing Demo</h1>
                <p>No LLM calls on this page — just a look at what LangGraph is remembering.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        steps = [
            ("1️⃣", "Every message is tied to a thread", "Your conversation is scoped to the **thread_id** shown below (also visible in the sidebar)."),
            ("2️⃣", "Every node writes a checkpoint", "After each node runs, LangGraph saves the full `State` to `checkpoints/checkpoints.sqlite`, keyed by that thread_id."),
            ("3️⃣", "The next turn loads it back", "Fields like `budget`, `source`, and `destination` are restored before any node runs — so nothing needs to be repeated."),
            ("4️⃣", "Restarts & interrupts resume cleanly", "If the app restarts, or the graph is paused on a human-approval `interrupt()`, resuming the same thread_id continues from exactly that checkpoint."),
        ]

        for icon, title, body in steps:
            with st.container(border=True):
                c1, c2 = st.columns([1, 12])
                c1.markdown(f"### {icon}")
                c2.markdown(f"**{title}**")
                c2.caption(body)

        st.write("")
        with st.container(border=True):
            st.markdown("#### 🧪 Try it yourself")
            st.markdown(
                "In the **Chat** tab, mention your budget in one message, then a few turns "
                "later ask *\"what transport should I choose?\"* — the assistant will use the "
                "budget without you repeating it."
            )

        st.write("")
        st.caption("Current session")
        st.markdown(
            f'<span class="thread-chip">🧵 thread_id: {st.session_state.thread_id}</span>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error("Something went wrong rendering this page.")
        st.exception(exc)
