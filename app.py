import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from ingest import load_pdf, split_into_chunks
from vectorstore import create_vectorstore, load_vectorstore, CHROMA_DIR
from rag_chain import build_rag_chain

load_dotenv()

# --- Page config — must be first Streamlit call ---
st.set_page_config(
    page_title="PeKA - A Personal Knowledge Assistant by Prasad Mokal",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Prasad's Personal PDF Chat Assistant")
st.caption("Upload a PDF and puchho jo puchna hai.")


# -------------------------------------------------------
# SIDEBAR — PDF upload and settings
# -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload the document you wanna analyse and chat about"
    )

    prompt_mode = st.selectbox(
    "Answer style",
    options=["default", "concise", "analyst", "german_tutor"],
    format_func=lambda x: {
        "default": "Default",
        "concise": "Concise",
        "analyst": "Analyst",
        "german_tutor": "🇩🇪 German Tutie for Cutie"   # nicer label in the dropdown
    }[x]
)

    process_btn = st.button("Process PDF", type="primary")


# -------------------------------------------------------
# PDF PROCESSING
# When user uploads a PDF and clicks "Process PDF"
# -------------------------------------------------------
if process_btn and uploaded_file:

    # Streamlit gives us the file as bytes in memory
    # PyPDFLoader needs a real file path on disk
    # So we save it to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name   # something like C:/Users/.../tmp12345.pdf

    with st.spinner("Reading and chunking PDF... bole to tukda tukda"):
        pages = load_pdf(tmp_path)
        chunks = split_into_chunks(pages)

    with st.spinner(f"Embedding {len(chunks)} chunks into vector store..."):
        vectorstore = create_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore

    with st.spinner("Building RAG chain... and someday.. a GOLD chain"):
        st.session_state.chain = build_rag_chain(vectorstore, prompt_mode)
        st.session_state.prompt_mode = prompt_mode
        st.session_state.chat_history = []   # reset chat on new PDF

    # Clean up the temp file
    os.unlink(tmp_path)

    st.sidebar.success(f"✅ Ready! {len(chunks)} chunks indexed.")


# -------------------------------------------------------
# If prompt mode changes after PDF is already processed,
# rebuild the chain without re-embedding
# -------------------------------------------------------
if ("chain" in st.session_state and
    st.session_state.get("prompt_mode") != prompt_mode):

    st.session_state.chain = build_rag_chain(
        st.session_state.vectorstore, prompt_mode
    )
    st.session_state.prompt_mode = prompt_mode


# -------------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------------

# Initialize chat history if it doesn't exist yet
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render all previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The chat input box at the bottom
question = st.chat_input(
    "Kya kaam hai jaldi bol...",
    disabled="chain" not in st.session_state  # greyed out until PDF is processed
)

if question:
    # Show user's message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Soch ne de bhai thoda..."):
            result = st.session_state.chain.invoke({"input": question})
            answer = result["answer"]

        st.markdown(answer)

        # Show sources in a collapsed section so they don't clutter the UI
        with st.expander("📄 Sources"):
            for i, doc in enumerate(result["context"]):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i+1} — Page {page}**")
                st.caption(doc.page_content[:300])

    # Save assistant answer to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# -------------------------------------------------------
# Empty state — shown before any PDF is uploaded
# -------------------------------------------------------
if "chain" not in st.session_state and not uploaded_file:
    st.info("👈 Idhar pdf upload karo nahi to mummy maaregi.")