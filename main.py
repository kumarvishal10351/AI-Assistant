import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil
import html
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────────────────────
# ENV & CONFIG
# ─────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["MISTRAL_API_KEY"]
    except:
        pass

st.set_page_config(
    page_title="Lexis · AI Book Assistant",
    page_icon="📖",
    layout="wide"
)

if not api_key:
    st.error("⚠️ MISTRAL_API_KEY not found.")
    st.stop()

os.environ["MISTRAL_API_KEY"] = api_key

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = os.path.exists("chroma_db")

if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "Precise"

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ts_now():
    return datetime.now().strftime("%H:%M")

def safe_text(text):
    return html.escape(text)

def safe_snippet(text, length=200):
    return html.escape(text[:length])

def score_doc(doc, q_words):
    return len(q_words & set(doc.page_content.lower().split()))

# ─────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return MistralAIEmbeddings()

@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.2)

@st.cache_resource
def get_vectorstore(_embed):
    if not os.path.exists("chroma_db"):
        return None
    return Chroma(persist_directory="chroma_db", embedding_function=_embed)
# SIDEBAR (FIXED — NO DUPLICATION)
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## 📖 Lexis AI")

    st.markdown("### 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    # 🔥 Answer Mode
    st.markdown("### 🧠 Answer Style")

    st.session_state.answer_mode = st.selectbox(
        "Mode",
        ["Precise", "Detailed", "Explain Like I'm 5"]
    )

    if uploaded_files:

        if st.button("⚙️ Build Knowledge Base"):

            with st.spinner("Indexing..."):

                all_docs = []
                embed = get_embeddings()

                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        path = tmp.name

                    loader = PyPDFLoader(path)
                    docs = loader.load()

                    for d in docs:
                        d.metadata["source"] = uf.name

                    all_docs.extend(docs)
                    os.unlink(path)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=300
                )

                chunks = splitter.split_documents(all_docs)

                Chroma.from_documents(
                    documents=chunks,
                    embedding=embed,
                    persist_directory="chroma_db"
                )

                get_vectorstore.clear()
                st.session_state.kb_ready = True

            st.success("✅ Knowledge Base Ready")

    st.markdown("---")

    if st.session_state.kb_ready:
        st.success("KB Ready")
    else:
        st.warning("Upload PDFs first")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if not st.session_state.kb_ready:
    st.info("Upload PDFs to begin")
    st.stop()

embed = get_embeddings()
llm = get_llm()
vectorstore = get_vectorstore(embed)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ─────────────────────────────────────────────────────────────
# PROMPT (UPGRADED)
# ─────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are Lexis AI.

Answer style: {st.session_state.answer_mode}

Rules:
- Precise → short answers
- Detailed → full explanation
- ELI5 → very simple

Use ONLY context.
Never hallucinate.
"""),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# ─────────────────────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask anything...")

if query:

    st.session_state.chat_history.append({"role": "user", "content": query})

    st.chat_message("user").write(query)

    # 🔥 MULTI QUERY RETRIEVAL
    mq_prompt = f"""
Generate 3 different search queries.

Question: {query}
"""
    queries = llm.invoke(mq_prompt).content.split("\n")

    all_docs = []
    for q in queries:
        if q.strip():
            all_docs.extend(retriever.invoke(q))

    docs = list({d.page_content: d for d in all_docs}.values())

    q_words = set(query.lower().split())
    docs = sorted(docs, key=lambda d: score_doc(d, q_words), reverse=True)[:4]

    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    with st.chat_message("assistant"):
        response = ""
        placeholder = st.empty()

        for chunk in llm.stream(final_prompt):
            response += chunk.content
            placeholder.write(response + "▌")

        placeholder.write(response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

    # 🔥 SOURCE TOGGLE BUTTON
    if st.button("📂 Show Sources"):
        st.session_state.show_sources = not st.session_state.show_sources

    if st.session_state.show_sources:
        st.markdown("### Sources")

        for d in docs:
            st.markdown(f"""
**📄 {d.metadata.get("source")} (Page {d.metadata.get("page","?")})**

_{safe_snippet(d.page_content)}..._
""")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Lexis · Mistral + ChromaDB RAG")