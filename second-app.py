from __future__ import annotations

import os

import streamlit as st

from ingest import CHUNKS_PATH, INDEX_DIR, REGISTRY_PATH, build_index
from rag_agent import MultiDocumentRAGAgent


def configure_openai_key() -> bool:
    secret_value = None
    try:
        secret_value = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret_value = None

    if secret_value:
        os.environ["OPENAI_API_KEY"] = secret_value

    return bool(os.environ.get("OPENAI_API_KEY"))


def index_exists() -> bool:
    return INDEX_DIR.exists() and CHUNKS_PATH.exists() and REGISTRY_PATH.exists()


@st.cache_resource(show_spinner=False)
def load_agent() -> MultiDocumentRAGAgent:
    return MultiDocumentRAGAgent()


def reset_agent_cache() -> None:
    load_agent.clear()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"**[{source['source_id']}] {source['domain_label']}**  \n`{source['file_name']}`")


st.set_page_config(page_title="Policy Assistant", layout="wide")
st.title("Policy Assistant")
st.caption("Ask questions across the indexed Arabic and English policy documents.")

has_key = configure_openai_key()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_rebuild_result" not in st.session_state:
    st.session_state.pending_rebuild_result = None

with st.sidebar:
    st.header("Knowledge Base")

    if not has_key:
        st.error("Set OPENAI_API_KEY in Streamlit secrets or your environment.")

    if index_exists():
        st.success("Ready")
    else:
        st.warning("Index has not been built yet")

    if st.button("Rebuild Knowledge Base", type="primary", disabled=not has_key):
        with st.spinner("Reading PDFs, embedding chunks, and saving the index..."):
            result = build_index()
            reset_agent_cache()
        st.session_state.pending_rebuild_result = result
        st.rerun()

    if st.session_state.pending_rebuild_result:
        result = st.session_state.pending_rebuild_result
        st.success(f"Indexed {result['document_count']} documents / {result['chunk_count']} chunks")

    st.divider()

    if st.button("New Conversation"):
        st.session_state.messages = []
        st.rerun()

if not has_key:
    st.stop()

if not index_exists():
    st.info("Click Rebuild Knowledge Base after placing PDF files in the documents folder.")
    st.stop()

try:
    with st.spinner("Loading knowledge base..."):
        agent = load_agent()
except Exception as exc:
    st.error(f"Could not load the RAG agent: {exc}")
    st.info("Try rebuilding the knowledge base from the sidebar after confirming dependencies and OPENAI_API_KEY are set.")
    st.stop()

registry = agent.registry

with st.sidebar:
    with st.expander("Indexed Documents", expanded=False):
        for doc in registry:
            st.markdown(
                f"**{doc['title']}**  \n"
                f"{doc['domain_label']} | {doc['language']} | {doc['chunk_count']} chunks"
            )

if not st.session_state.messages:
    st.info(
        "Ask a policy question, request a comparison, or ask for a summary. "
        "Answers include citations from the indexed documents."
    )

query = st.chat_input("Ask about any indexed policy, law, or control document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

latest_message_index = len(st.session_state.messages) - 1

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and index == latest_message_index:
            render_sources(message.get("sources", []))

should_answer = bool(st.session_state.messages and st.session_state.messages[-1]["role"] == "user")

if should_answer:
    active_query = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                result = agent.answer(active_query)
            except Exception as exc:
                st.error(f"Could not answer this question: {exc}")
                st.stop()

        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "route": result["route"],
        }
    )
