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
#  ENVIRONMENT & PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
# Fallback to Streamlit secrets (used on Streamlit Cloud)
if not api_key:
    try:
        api_key = st.secrets["MISTRAL_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

st.set_page_config(
    page_title="Lexis · AI Book Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CSS  — "Moonlit Library" aesthetic
#  Deep navy · Ivory · Soft violet glow · Playfair + Fira Mono
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Fira+Code:wght@400;500&family=Nunito:wght@400;500;600;700&display=swap');

/* ── CSS Variables ──────────────────────────────────────── */
:root{
    --navy:       #080d1a;
    --navy-mid:   #0d1425;
    --navy-card:  #111828;
    --navy-lift:  #161f32;
    --violet:     #7c6af7;
    --violet-dim: rgba(124,106,247,0.12);
    --violet-glow:rgba(124,106,247,0.06);
    --violet-bdr: rgba(124,106,247,0.22);
    --violet-hov: rgba(124,106,247,0.42);
    --teal:       #4ecdc4;
    --ivory:      #eae6dc;
    --ivory-dim:  #a09b8f;
    --border:     rgba(124,106,247,0.15);
    --border-soft:rgba(255,255,255,0.05);
    --user-grad:  linear-gradient(135deg,#1a1060,#120d45);
    --success:    #52c97a;
    --warning:    #f0a847;
    --danger:     #e05a6a;
}

/* ── Reset ──────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

html,body,[data-testid="stAppViewContainer"]{
    background:var(--navy) !important;
    font-family:'Nunito',sans-serif;
    color:var(--ivory);
}

/* Atmospheric background mesh */
[data-testid="stAppViewContainer"]::before{
    content:'';
    position:fixed;inset:0;
    background:
        radial-gradient(ellipse 60% 40% at 10% 0%,  rgba(124,106,247,0.07) 0%,transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 100%,rgba(78,205,196,0.05) 0%,transparent 55%),
        radial-gradient(ellipse 80% 60% at 50% 50%, rgba(8,13,26,1) 30%,transparent 100%);
    pointer-events:none;z-index:0;
}

[data-testid="stMain"]{background:transparent !important;}

.block-container{
    padding:1.2rem 1.8rem 5rem !important;
    max-width:940px !important;
    margin:0 auto;
}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"]{
    background:var(--navy-mid) !important;
    border-right:1px solid var(--border) !important;
}
section[data-testid="stSidebar"]>div{padding:1.2rem 0.9rem 2rem;}

/* ── Sidebar brand block ────────────────────────────────── */
.sb-brand{
    text-align:center;
    padding:0.5rem 0 1.3rem;
    border-bottom:1px solid var(--border-soft);
    margin-bottom:1.1rem;
}
.sb-logo{
    font-size:2.2rem;
    margin-bottom:6px;
    display:block;
    filter:drop-shadow(0 0 14px rgba(124,106,247,0.45));
}
.sb-brand h2{
    font-family:'Playfair Display',serif;
    font-weight:700;
    font-size:1.3rem;
    color:var(--ivory);
    letter-spacing:0.03em;
    margin-bottom:3px;
}
.sb-brand p{
    font-family:'Fira Code',monospace;
    font-size:0.58rem;
    color:var(--violet);
    letter-spacing:0.18em;
    text-transform:uppercase;
    opacity:0.75;
}

/* ── Sidebar labels ─────────────────────────────────────── */
.sb-label{
    font-family:'Fira Code',monospace;
    font-size:0.58rem;
    letter-spacing:0.2em;
    text-transform:uppercase;
    color:rgba(124,106,247,0.55);
    margin:1rem 0 0.4rem;
    padding-left:2px;
}

/* ── File pills ─────────────────────────────────────────── */
.file-pill{
    display:flex;align-items:center;gap:7px;
    background:var(--violet-dim);
    border:1px solid var(--border);
    border-radius:8px;
    padding:6px 10px;
    margin-bottom:5px;
    font-size:0.76rem;color:var(--ivory);
    word-break:break-all;line-height:1.3;
}
.file-pill .fi{color:var(--violet);flex-shrink:0;font-size:1rem;}

/* ── KB status badge ────────────────────────────────────── */
.kb-badge{
    display:flex;align-items:center;gap:8px;
    background:var(--navy-card);
    border:1px solid var(--border);
    border-radius:8px;
    padding:8px 11px;
    font-family:'Fira Code',monospace;
    font-size:0.66rem;
    letter-spacing:0.06em;
    margin-bottom:8px;
}
.kb-badge .dot{
    width:7px;height:7px;
    border-radius:50%;
    flex-shrink:0;
}
.dot-green{background:var(--success);box-shadow:0 0 6px var(--success);}
.dot-orange{background:var(--warning);box-shadow:0 0 6px var(--warning);}

/* ── Buttons ────────────────────────────────────────────── */
.stButton>button{
    width:100% !important;
    background:var(--violet-dim) !important;
    border:1px solid var(--border) !important;
    color:var(--ivory) !important;
    border-radius:10px !important;
    font-family:'Nunito',sans-serif !important;
    font-weight:600 !important;
    font-size:0.82rem !important;
    padding:0.52rem 1rem !important;
    margin-bottom:5px !important;
    transition:all 0.2s ease !important;
    text-align:left !important;
}
.stButton>button:hover{
    background:rgba(124,106,247,0.22) !important;
    border-color:var(--violet-hov) !important;
    color:#c8c0ff !important;
    box-shadow:0 0 18px rgba(124,106,247,0.12) !important;
    transform:translateX(2px) !important;
}

/* ── File uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"]{
    background:var(--navy-card) !important;
    border:1px dashed var(--border) !important;
    border-radius:12px !important;
}
[data-testid="stFileUploader"]:hover{
    border-color:var(--violet-hov) !important;
}

/* ── Selectbox ──────────────────────────────────────────── */
[data-testid="stSelectbox"]>div>div{
    background:var(--navy-card) !important;
    border:1px solid var(--border) !important;
    border-radius:10px !important;
    color:var(--ivory) !important;
    font-family:'Nunito',sans-serif !important;
    font-size:0.86rem !important;
}

/* ── Page header ────────────────────────────────────────── */
.page-header{
    display:flex;align-items:center;gap:14px;
    padding:1.1rem 1.6rem;
    background:var(--navy-card);
    border:1px solid var(--border);
    border-radius:16px;
    margin-bottom:1.4rem;
    position:relative;overflow:hidden;
}
.page-header::before{
    content:'';
    position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,var(--violet),transparent);
    opacity:0.5;
}
.ph-icon{
    font-size:2rem;
    filter:drop-shadow(0 0 10px rgba(124,106,247,0.4));
    flex-shrink:0;
}
.ph-title{
    font-family:'Playfair Display',serif;
    font-weight:700;font-size:1.5rem;
    color:var(--ivory);letter-spacing:0.02em;
    line-height:1;margin-bottom:4px;
}
.ph-sub{
    font-family:'Fira Code',monospace;
    font-size:0.6rem;color:var(--violet);
    letter-spacing:0.15em;text-transform:uppercase;opacity:0.75;
}
.ph-right{margin-left:auto;}
.status-pill{
    display:flex;align-items:center;gap:7px;
    background:rgba(82,201,122,0.09);
    border:1px solid rgba(82,201,122,0.22);
    border-radius:20px;padding:4px 13px;
    font-family:'Fira Code',monospace;
    font-size:0.6rem;color:var(--success);
    letter-spacing:0.12em;white-space:nowrap;
}
.pulse{
    width:6px;height:6px;background:var(--success);
    border-radius:50%;box-shadow:0 0 6px var(--success);
    animation:pulse-anim 2s ease-in-out infinite;
}
@keyframes pulse-anim{0%,100%{opacity:1;}50%{opacity:0.25;}}

/* ── Section heading ────────────────────────────────────── */
.sec-head{
    display:flex;align-items:center;gap:10px;
    font-family:'Playfair Display',serif;
    font-style:italic;font-size:0.95rem;
    color:rgba(124,106,247,0.85);
    margin:1.3rem 0 0.7rem;
}
.sec-head::after{
    content:'';flex:1;height:1px;
    background:var(--border);border-radius:2px;
}

/* ── Divider ────────────────────────────────────────────── */
.vdivider{
    border:none;height:1px;
    background:linear-gradient(90deg,transparent,var(--border),transparent);
    margin:0.9rem 0;
}

/* ── Chat bubbles ───────────────────────────────────────── */
.chat-wrap{display:flex;flex-direction:column;gap:14px;padding:0.4rem 0;}

.msg-row{display:flex;align-items:flex-end;gap:10px;
    animation:slidein 0.28s ease;}
.msg-row.user{justify-content:flex-end;}
.msg-row.bot {justify-content:flex-start;}

@keyframes slidein{
    from{opacity:0;transform:translateY(10px);}
    to  {opacity:1;transform:translateY(0);}
}

.av{
    width:34px;height:34px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:15px;flex-shrink:0;
}
.av-user{
    background:linear-gradient(135deg,#2d1f8a,#1a1060);
    border:1px solid rgba(124,106,247,0.35);
    order:2;
}
.av-bot{
    background:linear-gradient(135deg,#161f32,#0d1425);
    border:1px solid var(--border);
    box-shadow:0 0 10px rgba(124,106,247,0.15);
}

.bubble{max-width:76%;padding:12px 16px;font-size:0.9rem;
    line-height:1.65;border-radius:16px;white-space:pre-wrap;word-break:break-word;}
.bubble.user{
    background:var(--user-grad);
    color:var(--ivory);
    border:1px solid rgba(124,106,247,0.25);
    border-radius:16px 16px 4px 16px;
    box-shadow:0 4px 22px rgba(0,0,0,0.35);
}
.bubble.bot{
    background:var(--navy-card);
    color:var(--ivory);
    border:1px solid var(--border);
    border-radius:16px 16px 16px 4px;
    box-shadow:0 4px 22px rgba(0,0,0,0.4);
}
.bubble .ts{
    display:block;
    font-family:'Fira Code',monospace;
    font-size:0.56rem;color:rgba(160,155,143,0.5);
    margin-top:7px;text-align:right;
}

/* ── Welcome state ──────────────────────────────────────── */
.welcome{
    text-align:center;
    padding:3rem 1rem 2rem;
}
.welcome .wi{font-size:3.5rem;margin-bottom:1rem;
    display:block;
    filter:drop-shadow(0 0 18px rgba(124,106,247,0.3));}
.welcome h3{
    font-family:'Playfair Display',serif;
    font-weight:700;font-size:1.5rem;
    color:var(--ivory);margin-bottom:8px;
}
.welcome p{font-size:0.88rem;color:var(--ivory-dim);
    max-width:380px;margin:0 auto;line-height:1.65;}
.hint-row{
    display:flex;flex-wrap:wrap;justify-content:center;
    gap:8px;margin-top:1.3rem;
}
.hint{
    background:var(--violet-dim);
    border:1px solid var(--border);
    color:rgba(160,155,143,0.75);
    padding:5px 13px;border-radius:20px;
    font-size:0.77rem;
}

/* ── Confidence bar ─────────────────────────────────────── */
.conf-wrap{
    background:var(--navy-card);
    border:1px solid var(--border);
    border-radius:12px;
    padding:13px 16px;margin-bottom:1rem;
}
.conf-lbl{
    font-family:'Fira Code',monospace;
    font-size:0.6rem;letter-spacing:0.15em;
    text-transform:uppercase;
    color:rgba(124,106,247,0.65);margin-bottom:9px;
}
.conf-track{
    background:rgba(255,255,255,0.05);
    border-radius:6px;height:7px;overflow:hidden;
}
.conf-fill{height:100%;border-radius:6px;transition:width 0.7s ease;}
.conf-val{
    font-family:'Playfair Display',serif;
    font-weight:700;font-size:1.1rem;
    margin-top:7px;
}

/* ── Source cards ───────────────────────────────────────── */
.src-card{
    background:var(--navy-card);
    border:1px solid var(--border);
    border-left:3px solid var(--violet);
    border-radius:0 11px 11px 0;
    padding:12px 15px;margin-bottom:8px;
    transition:border-color 0.2s,box-shadow 0.2s;
}
.src-card:hover{
    border-color:var(--violet-hov);
    box-shadow:0 0 20px rgba(124,106,247,0.08);
}
.src-name{
    font-family:'Playfair Display',serif;
    font-size:0.83rem;color:#b3acff;
    font-weight:500;margin-bottom:3px;
}
.src-meta{
    font-family:'Fira Code',monospace;
    font-size:0.6rem;color:var(--ivory-dim);
    letter-spacing:0.06em;margin-bottom:7px;opacity:0.7;
}
.src-snip{
    font-size:0.8rem;color:#7a7060;
    line-height:1.55;font-style:italic;
}

/* ── Chat input override ────────────────────────────────── */
[data-testid="stChatInput"]{
    background:var(--navy-card) !important;
    border:1px solid var(--border) !important;
    border-radius:14px !important;
    box-shadow:0 0 0 0 transparent !important;
    transition:border-color 0.2s,box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within{
    border-color:var(--violet-hov) !important;
    box-shadow:0 0 28px rgba(124,106,247,0.1) !important;
}
[data-testid="stChatInput"] textarea{
    font-family:'Nunito',sans-serif !important;
    font-size:0.9rem !important;
    color:var(--ivory) !important;
    background:transparent !important;
}

/* ── Spinner ────────────────────────────────────────────── */
[data-testid="stSpinner"] p{
    font-family:'Fira Code',monospace !important;
    font-size:0.76rem !important;
    color:var(--violet) !important;
    letter-spacing:0.08em;
}

/* ── Alerts ─────────────────────────────────────────────── */
[data-testid="stAlert"]{
    background:var(--navy-card) !important;
    border:1px solid var(--border) !important;
    border-radius:10px !important;
    color:var(--ivory) !important;
}

/* ── Footer ─────────────────────────────────────────────── */
.footer{
    text-align:center;margin-top:1.5rem;
    padding-top:0.8rem;
    border-top:1px solid rgba(255,255,255,0.04);
}
.footer span{
    font-family:'Fira Code',monospace;
    font-size:0.58rem;
    color:rgba(100,100,130,0.4);
    letter-spacing:0.14em;text-transform:uppercase;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(124,106,247,0.2);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(124,106,247,0.4);}

/* ── Hide Streamlit chrome ──────────────────────────────── */
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stDecoration"]{display:none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  GUARD: API KEY
# ─────────────────────────────────────────────────────────────
if not api_key:
    st.error("⚠️ MISTRAL_API_KEY not found. Add it to your .env file (local) or Streamlit Cloud Secrets (deployed).")
    st.stop()

os.environ["MISTRAL_API_KEY"] = api_key

# ─────────────────────────────────────────────────────────────
#  SESSION STATE  (initialize once)
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # [{role, content, time}]
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = os.path.exists("chroma_db")

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def ts_now() -> str:
    return datetime.now().strftime("%H:%M")

def score_doc(doc, q_words: set) -> int:
    return len(q_words & set(doc.page_content.lower().split()))

def confidence_score(docs, rewritten_query: str) -> int:
    if not docs:
        return 0
    q_words = set(rewritten_query.lower().split())
    hits    = sum(score_doc(d, q_words) for d in docs)
    ratio   = min(hits / max(len(q_words), 1), 1.0)
    return min(int(ratio * 70) + min(len(docs) * 8, 30), 100)

def safe_snippet(text: str, length: int = 220) -> str:
    return html.escape(text[:length])

def safe_text(text: str) -> str:
    return html.escape(text)

# ─────────────────────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.2)

@st.cache_resource(show_spinner=False)
def get_vectorstore(_embeddings):
    if not os.path.exists("chroma_db"):
        return None
    return Chroma(persist_directory="chroma_db", embedding_function=_embeddings)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">
        <span class="sb-logo">📖</span>
        <h2>Lexis</h2>
        <p>AI Book Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────
    st.markdown('<p class="sb-label">📂 Upload Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            st.markdown(
                f'<div class="file-pill"><span class="fi">📄</span>{safe_text(f.name)}</div>',
                unsafe_allow_html=True,
            )

        if st.button("⚙️  Build Knowledge Base"):
            with st.spinner("📚 Indexing documents…"):
                all_docs   = []
                embeddings = get_embeddings()

                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs   = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = uf.name
                        all_docs.extend(docs)
                    finally:
                        os.unlink(tmp_path)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200, chunk_overlap=300
                )
                chunks = splitter.split_documents(all_docs)

                if os.path.exists("chroma_db"):
                    vs = Chroma(
                        persist_directory="chroma_db",
                        embedding_function=embeddings,
                    )
                    vs.add_documents(chunks)
                else:
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory="chroma_db",
                    )

                get_vectorstore.clear()          # refresh cached store
                st.session_state.kb_ready = True

            st.success(f"✅ {len(chunks)} chunks indexed from {len(uploaded_files)} file(s).")

    # ── KB status ────────────────────────────────────────────
    st.markdown('<hr class="vdivider">', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">🗄 Knowledge Base</p>', unsafe_allow_html=True)

    if st.session_state.kb_ready:
        st.markdown("""
        <div class="kb-badge">
            <div class="dot dot-green"></div>
            <span style="color:#52c97a;">Database loaded</span>
        </div>""", unsafe_allow_html=True)

        if st.button("🗑  Reset Knowledge Base"):
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")
            get_vectorstore.clear()
            st.session_state.kb_ready       = False
            st.session_state.chat_history   = []
            st.rerun()
    else:
        st.markdown("""
        <div class="kb-badge">
            <div class="dot dot-orange"></div>
            <span style="color:#f0a847;">No database — upload PDFs</span>
        </div>""", unsafe_allow_html=True)

    # ── Chat controls ────────────────────────────────────────
    st.markdown('<hr class="vdivider">', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">💬 Conversation</p>', unsafe_allow_html=True)

    if st.button("🔄  Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # ── Model info ───────────────────────────────────────────
    st.markdown('<hr class="vdivider">', unsafe_allow_html=True)
    st.markdown("""
    <p class="sb-label" style="margin-top:0;">⚙️ Model Config</p>
    <p style="font-family:'Fira Code',monospace;font-size:0.62rem;
              color:#3d3a50;line-height:1.9;margin:0;">
        LLM &nbsp;· mistral-small-2506<br>
        Embed · mistral-embed<br>
        Store · ChromaDB (MMR k=6)<br>
        Chunk · 1200 / overlap 300
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="ph-icon">📖</div>
    <div>
        <div class="ph-title">Lexis</div>
        <div class="ph-sub">Multi-Document AI Book Assistant · RAG + Mistral</div>
    </div>
    <div class="ph-right">
        <div class="status-pill"><div class="pulse"></div>ONLINE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  GATE: no KB yet
# ─────────────────────────────────────────────────────────────
if not st.session_state.kb_ready:
    st.markdown("""
    <div class="welcome">
        <span class="wi">📚</span>
        <h3>Welcome to Lexis</h3>
        <p>Upload your PDF books or documents in the sidebar,
           then click <strong>Build Knowledge Base</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────────────────────
embeddings  = get_embeddings()
llm         = get_llm()
vectorstore = get_vectorstore(embeddings)

if vectorstore is None:
    st.warning("Knowledge base folder missing. Please rebuild from the sidebar.")
    st.stop()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)

# ─────────────────────────────────────────────────────────────
#  DOCUMENT FILTER
# ─────────────────────────────────────────────────────────────
raw_meta    = vectorstore.get()["metadatas"]
all_sources = sorted({m.get("source", "Unknown") for m in raw_meta if m})

col_sel, _ = st.columns([3, 2])
with col_sel:
    selected_doc = st.selectbox(
        "🔖 Filter by document",
        ["All Documents"] + all_sources,
    )


# ─────────────────────────────────────────────────────────────
#  PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Lexis, a precise and insightful literary AI assistant.
Answer questions using ONLY the provided document context.

Guidelines:
- Synthesise information across multiple passages when relevant.
- Be accurate, clear, and well-structured.
- If the answer is not in the context, say so honestly — never fabricate.
- Reference specific documents or pages when it strengthens the answer.
- Write in clean, readable paragraphs."""),

    ("human", """Conversation so far:
{chat_history}

Document context:
{context}

Question:
{question}"""),
])


# ─────────────────────────────────────────────────────────────
#  DISPLAY CHAT HISTORY  (rendered before the input widget)
# ─────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome">
        <span class="wi">🔍</span>
        <h3>Ask anything about your books</h3>
        <p>Your knowledge base is ready. Type a question below to begin.</p>
        <div class="hint-row">
            <span class="hint">📖 Summarise a chapter</span>
            <span class="hint">🔍 Find key concepts</span>
            <span class="hint">💡 Compare ideas</span>
            <span class="hint">🗂 List main themes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role = msg["role"]
        text = safe_text(msg["content"])
        ts   = msg.get("time", "")
        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="bubble user">{text}<span class="ts">{ts}</span></div>
                <div class="av av-user">👤</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="av av-bot">📖</div>
                <div class="bubble bot">{text}<span class="ts">{ts}</span></div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CHAT INPUT
#
#  WHY st.chat_input and NOT st.text_input:
#  st.text_input keeps its value across reruns, so after we call
#  st.rerun() the widget still holds the old query and the block
#  fires again → infinite loop.
#  st.chat_input clears itself automatically after each submit,
#  so the `if query:` block only fires ONCE per user message.
#  We also never call st.rerun() after processing — not needed.
# ─────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about your documents…")

if query:

    # 1. Save & show the user bubble immediately
    st.session_state.chat_history.append(
        {"role": "user", "content": query, "time": ts_now()}
    )
    st.markdown(f"""
    <div class="chat-wrap">
        <div class="msg-row user">
            <div class="bubble user">
                {safe_text(query)}<span class="ts">{ts_now()}</span>
            </div>
            <div class="av av-user">👤</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # 2. Build conversation history string (last 6 prior turns, excluding current message)
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.chat_history[:-1][-6:]
    )

    # 3. Rewrite query for better retrieval
    with st.spinner("🔎 Understanding your question…"):
        rewrite_prompt = (
            "Rewrite the following question to be clear and specific for "
            "semantic document retrieval. Output ONLY the rewritten question.\n\n"
            f"History:\n{history_text}\n\n"
            f"Original question: {query}"
        )
        rewritten = llm.invoke(rewrite_prompt).content.strip()

    # 4. Retrieve & filter
    with st.spinner("📚 Searching knowledge base…"):
        docs = retriever.invoke(rewritten)

        if selected_doc != "All Documents":
            docs = [d for d in docs if d.metadata.get("source") == selected_doc]

        q_words = set(rewritten.lower().split())
        docs    = sorted(docs, key=lambda d: score_doc(d, q_words), reverse=True)[:4]

    if not docs:
        st.warning(
            "⚠️ No relevant passages found. "
            "Try rephrasing, or switch the filter to **All Documents**."
        )
        # Remove the unanswered user message so it doesn't linger
        st.session_state.chat_history.pop()
        st.stop()

    # 5. Build context string
    context = "\n\n".join(
        f"[{d.metadata.get('source','Unknown')} · Page {d.metadata.get('page','?')}]\n"
        f"{d.page_content}"
        for d in docs
    )

    # 6. Generate answer
    final_prompt = prompt.invoke({
        "context":      context,
        "question":     query,
        "chat_history": history_text,
    })

    st.markdown('<div class="sec-head">Answer</div>', unsafe_allow_html=True)
    placeholder = st.empty()
    full_text   = ""

    # Real streaming — preserves all formatting (newlines, paragraphs, lists)
    # Throttle UI updates to every ~15 chars to avoid excessive re-renders/flickering
    _last_render = 0
    for chunk in llm.stream(final_prompt):
        full_text += chunk.content
        if len(full_text) - _last_render >= 15:
            _last_render = len(full_text)
            placeholder.markdown(
                f'<div class="bubble bot" style="max-width:100%;">'
                f'{safe_text(full_text)}<span style="opacity:0.35;">▌</span></div>',
                unsafe_allow_html=True,
            )

    final_answer = full_text.strip()
    placeholder.markdown(
        f'<div class="bubble bot" style="max-width:100%;">{safe_text(final_answer)}</div>',
        unsafe_allow_html=True,
    )

    # 7. Save assistant reply to history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_answer, "time": ts_now()}
    )

    # 8. Confidence meter
    conf = confidence_score(docs, rewritten)
    if conf >= 70:
        fill = "linear-gradient(90deg,#2d8a52,#52c97a)"; vc = "#52c97a"
    elif conf >= 40:
        fill = "linear-gradient(90deg,#a06010,#f0a847)"; vc = "#f0a847"
    else:
        fill = "linear-gradient(90deg,#8a2030,#e05a6a)"; vc = "#e05a6a"

    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-lbl">Retrieval Confidence</div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{conf}%;background:{fill};
                box-shadow:0 0 8px {vc}55;"></div>
        </div>
        <div class="conf-val" style="color:{vc};">{conf}%</div>
    </div>
    """, unsafe_allow_html=True)

    # 9. Source cards
    st.markdown('<div class="sec-head">Sources Used</div>', unsafe_allow_html=True)
    for doc in docs:
        src     = safe_text(doc.metadata.get("source", "Unknown"))
        page    = doc.metadata.get("page", "?")
        snippet = safe_snippet(doc.page_content)
        st.markdown(f"""
        <div class="src-card">
            <div class="src-name">📄 {src}</div>
            <div class="src-meta">Page {page}</div>
            <div class="src-snip">"{snippet}…"</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Lexis · Mistral AI + ChromaDB RAG · Local Intelligence</span>
</div>
""", unsafe_allow_html=True)