import os
import hashlib
import time
import streamlit as st

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---- Import your pipeline ----
from rag_pipeline.loader import load_pdf
from rag_pipeline.splitter import split_documents
from rag_pipeline.embedding import get_embeddings
from rag_pipeline.vectorstore import create_vectorstore
from rag_pipeline.retriever import get_retriever
from rag_pipeline.qa_chain import build_chain

# ---- Page ----
st.set_page_config(page_title="⚡ AI Document Assistant", layout="wide")
st.title("⚡ AI Document Assistant")

# ---- Session ----
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# =========================================================
# 🔥 Streaming (simple + correct)
# =========================================================
def render_streaming_response(text):
    placeholder = st.empty()
    full_text = ""

    for char in text:
        full_text += char
        placeholder.markdown(full_text + "▌")
        time.sleep(0.003)

    placeholder.markdown(full_text)
    return full_text


# =========================================================
# PIPELINE
# =========================================================
@st.cache_resource
def build_pipeline(file_path, file_hash):

    persist_dir = f"db/{file_hash}"
    embedding = get_embeddings()

    if os.path.exists(persist_dir):
        db = create_vectorstore(None, embedding, file_hash)
    else:
        docs = load_pdf(file_path)
        chunks = split_documents(docs)
        db = create_vectorstore(chunks, embedding, file_hash)

    retriever = get_retriever(db)
    qa_chain = build_chain(retriever)

    return qa_chain


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        file_bytes = uploaded_file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        file_path = f"temp_{file_hash}.pdf"

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        if st.session_state.processed_file != file_hash:

            with st.spinner("Processing document..."):
                qa_chain = build_pipeline(file_path, file_hash)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed_file = file_hash

            st.success("✅ Ready!")

        else:
            st.success("⚡ Already processed")


# =========================================================
# CHAT UI (FIXED — no duplication)
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Ask anything..."):

    # show user instantly
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({
                "question": user_input
            })

        full_response = render_streaming_response(response)

    # save ONLY once
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })