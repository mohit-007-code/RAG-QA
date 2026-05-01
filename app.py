import os
import time
import re
import hashlib
import streamlit as st

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---- Imports ----
from rag_pipeline.loader import load_pdf
from rag_pipeline.splitter import split_documents
from rag_pipeline.embedding import get_embeddings
from rag_pipeline.vectorstore import create_vectorstore
from rag_pipeline.retriever import get_retriever
from rag_pipeline.qa_chain import build_qa_chain

# ---- Page config ----
st.set_page_config(page_title="AI Document Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Document Assistant")
st.caption("Ask questions from your uploaded PDF")

# ---- Session State ----
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# =========================================================
# 🔥 CACHE RAG PIPELINE (FASTER)
# =========================================================
@st.cache_resource
def build_rag_pipeline(file_path, file_hash):
    docs = load_pdf(file_path)
    chunks = split_documents(docs)

    embedding = get_embeddings()
    db = create_vectorstore(chunks, embedding, file_hash) 

    retriever = get_retriever(db)
    qa_chain = build_qa_chain(retriever)

    return retriever, qa_chain


# =========================================================
# 🚀 RESPONSE FORMATTER (IMPROVED)
# =========================================================
def format_response(text):
    # Remove junk tokens
    text = re.sub(r"(System:|Human:|Assistant:)", "", text)

    # Normalize spacing
    text = text.strip()

    # Fix numbering
    text = re.sub(r"(\d+)\.\s*", r"\n\n\1. ", text)

    # Fix bullets
    text = re.sub(r"\n\s*-\s*", "\n- ", text)

    # Detect code more reliably
    if is_code_response(text):
        code = extract_code(text)
        explanation = text.replace(code, "").strip()

        return f"{explanation}\n\n```python\n{code}\n```"

    return text


def is_code_response(text):
    keywords = [
        "def ", "class ", "import ", "return ",
        "for ", "while ", "if ", "print("
    ]
    return sum(k in text for k in keywords) >= 2


def extract_code(text):
    lines = text.split("\n")
    code_lines = []

    for line in lines:
        if any(k in line for k in [
            "def ", "class ", "import ", "return ",
            "for ", "while ", "if ", "print(", "="
        ]):
            code_lines.append(line)

    return "\n".join(code_lines)


# =========================================================
# 📂 SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        file_bytes = uploaded_file.read()

        # 🔥 Create file hash (prevents reprocessing same file)
        file_hash = hashlib.md5(file_bytes).hexdigest()

        file_path = f"temp_{file_hash}.pdf"

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        if st.session_state.processed_file != file_hash:
            with st.spinner("Processing document..."):
                retriever, qa_chain = build_rag_pipeline(file_path, file_hash)

                st.session_state.retriever = retriever
                st.session_state.qa_chain = qa_chain
                st.session_state.processed_file = file_hash

            st.success("✅ Document processed!")

        else:
            st.success("⚡ Using cached document")

    st.markdown("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []


# =========================================================
# 💬 DISPLAY CHAT
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# 💬 CHAT INPUT
# =========================================================
if user_input := st.chat_input("Ask something about your document..."):

    if len(user_input.strip()) < 3:
        st.warning("⚠️ Ask a meaningful question.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.qa_chain:

        with st.chat_message("assistant"):

            placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke({
                    "question": user_input
                })

            # 🚀 FORMAT RESPONSE
            response = format_response(response)

            if response.strip().lower() == "i don't know":
                response = "⚠️ Answer not found in document."

            # 🚀 SMOOTHER STREAMING
            for chunk in response.split(" "):
                full_response += chunk + " "
                placeholder.markdown(full_response)
                time.sleep(0.01)

            # ---- SOURCE ----
            with st.expander("📄 Source Context"):
                docs = st.session_state.retriever.invoke(user_input)
                for d in docs:
                    st.write(d.page_content[:400])

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

    else:
        st.warning("⚠️ Upload a PDF first.")