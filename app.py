import os
import uuid
import time
import threading
import streamlit as st
import queue
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from logging_utils import setup_logging
from validate import validate_user_text
from global_limit import check_rate_limit_global
from knowledge_base import KnowledgeBase
from create_agent import run_agent, get_agent

st.set_page_config(page_title="Investment Portfolio AI Manager", layout="wide")
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
logger = setup_logging(level="INFO", name="etf_rag_bot")

KB_URLS = [
    "https://rpc.cfainstitute.org/sites/default/files/-/media/documents/book/rf-publication/2015/rf-v2015-n3-1-pdf.pdf",
    "https://www.vanguardsouthamerica.com/content/dam/intl/americas/documents/latam/en/2022/08/mx-sa-2331724-portfolio-construction-framework.pdf",
    "https://www.eib.org/files/efs/economics_working_paper_2019_11_en.pdf",
    "https://www.federalreserve.gov/econres/ifdp/files/ifdp1222r1.pdf",
    "https://www.msci.com/documents/1296102/1336482/Foundations_of_Factor_Investing.pdf",
    "https://www.nber.org/system/files/working_papers/w10080/w10080.pdf",
    "https://obj.portfolioconstructionforum.edu.au/articles_perspectives/Portfolio-Construction-Forum_DG_Equity-factor-based-investing-a-practitioners-guide.PDF",
]


def get_openai_key() -> str:
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (Streamlit secrets or env var).")
    return key


@st.cache_resource
def get_kb():
    return KnowledgeBase(
        urls=KB_URLS,
        open_ai_key=get_openai_key(),
        index_dir="kb_index",
        chunk_size=600,
        chunk_overlap=100,
    )


@st.cache_resource
def get_agent_main():
    return get_agent()


agent = get_agent_main()
thread_id = st.session_state['thread_id']


class QueueTokenCallback(BaseCallbackHandler):
    """Collect tokens in a thread-safe manner without utilizing Streamlit."""

    def __init__(self, token_queue: "queue.Queue[str]"):
        self.queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs):
        if token:
            self.queue.put(token)


def run_agent_streaming(history_for_model, tool_callback, agent, thread_id, placeholder):
    """Wrapper around `run_agent` to safely stream response tokens"""
    token_queue: queue.Queue[str] = queue.Queue()
    result_queue: queue.Queue[str] = queue.Queue()

    token_callback = QueueTokenCallback(token_queue)

    def worker_thread():
        try:
            answer = run_agent(history_for_model, [token_callback, tool_callback], agent=agent, thread_id=thread_id)
            result_queue.put(('ok', answer))
        except Exception as e:
            result_queue.put(('err', e))

    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()

    text = ""

    while t.is_alive():
        try:
            while True:
                text += token_queue.get_nowait()
        except queue.Empty:
            pass
        placeholder.markdown(text + "▌")
        time.sleep(0.02)

    while not (token_queue.qsize() == 0):
        text += token_queue.get()

    placeholder.markdown(text)

    status, payload = result_queue.get()
    if status == 'err':
        raise payload
    if text.strip():
        return text
    if payload is None:
        return ""
    return str(payload)


class ToolTraceCallback(BaseCallbackHandler):
    """Collects tool calls and outputs for display."""

    def __init__(self):
        self.events = []

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        name = serialized.get("name", "unknown_tool")
        self.events.append({"tool": name, "status": "start", "input": (input_str or "")[:800]})

    def on_tool_end(self, output, **kwargs):
        self.events.append({"status": "end", "output": str(output)[:1200]})

    def on_tool_error(self, error, **kwargs):
        self.events.append({"status": "error", "error": str(error)[:1200]})


st.title("💬 ETF RAG Chatbot (RAG + Tools + MCP)")
st.caption("Educational only. No financial advice. Uses KB retrieval + tools + Alpha Vantage MCP for single stocks.")

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Knowledge Base Reference Count", 1, 10, 4)
    history_n = st.slider("Chat history messages", 4, 20, 12)
    show_debug = st.checkbox("Show KB sources + tool traces", value=True)
    st.divider()
    st.write("**Rate limit:** 20 requests / 10 minutes (global)")

# Session identifiers
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("client_id", st.session_state["thread_id"])

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = [
        AIMessage(content="Hi! Ask me about portfolio allocation, risk reports, macro regime, or stocks.")
    ]

# Render chat history
for msg in st.session_state.chat:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)


kb = get_kb()
if "kb_ready" not in st.session_state:
    with st.spinner("Preparing knowledge base (build/load)…"):
        # build_or_load is async, but your KB class supports it :contentReference[oaicite:10]{index=10}
        import asyncio
        try:
            asyncio.run(kb.build_or_load())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(kb.build_or_load())
            loop.close()
    st.session_state.kb_ready = True

user_input = st.chat_input("Type your message…")

if user_input:
    t0 = time.time()

    # 1) Validate input
    try:
        user_text = validate_user_text(user_input)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # 2) Global rate limiting (Firestore)
    allowed, retry_after = check_rate_limit_global(st.session_state["client_id"])
    if not allowed:
        st.error(f"Rate limit reached. Try again in ~{retry_after}s.")
        logger.warning("rate_limited client_id=%s retry_after=%s", st.session_state["client_id"], retry_after)
        st.stop()

    # 3) Append user message to history
    st.session_state.chat.append(HumanMessage(content=user_text))
    with st.chat_message("user"):
        st.write(user_text)

    # 4) Retrieve KB context every turn
    with st.status("Retrieving knowledge base context…", expanded=False):
        kb_context, kb_sources = kb.retrieve(user_text, k=k)

    augmented_user_text = f"""
    KB CONTEXT (may be irrelevant; reference only):
    {kb_context}

    INSTRUCTIONS:
    - First decide if KB context is relevant.
    - If relevant: apply KB rules.
    - If not: ignore KB and state "KB not used".

    USER QUESTION:
    {user_text}
    """.strip()

    # 5) Prepare agent input messages (limit history)
    history = st.session_state.chat[-history_n:]
    history_for_model = history[:-1] + [HumanMessage(content=augmented_user_text)]

    # 6) Run agent with streaming + tool tracing
    with st.chat_message("assistant"):
        tool_callback = ToolTraceCallback()
        placeholder = st.empty()
        status_box = st.status("Running agent (news -> tools -> answer)…", expanded=False)
        try:
            answer = run_agent_streaming(history_for_model, tool_callback, agent, thread_id,
                                         placeholder)
            status_box.update(label="Done!", state='complete', expanded=False)
            st.session_state.chat.append(AIMessage(content=answer))

            if show_debug:
                with st.expander("🔎 Tool calls (this turn)"):
                    st.json(tool_callback.events)

                with st.expander("📚 KB sources (this turn)"):
                    st.json(kb_sources)

        except Exception:
            logger.exception("agent_run_failed")
            st.error("Agent failed. Check logs for details.")

    logger.info("turn_complete ms=%d", int((time.time() - t0) * 1000))
