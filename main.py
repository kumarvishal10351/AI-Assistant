import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ================= SETUP =================
load_dotenv()
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("API key not found. Please check your .env file")
    st.stop()

os.environ["MISTRAL_API_KEY"] = api_key

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Assistant Pro", page_icon="🤖", layout="wide")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Settings")

if st.sidebar.button("🧹 Reset Knowledge Base"):
    if os.path.exists("chroma_db"):
        import shutil
        shutil.rmtree("chroma_db")
        st.sidebar.success("Database cleared!")

# ================= UI =================
st.title("📚 AI Assistant")
st.caption("Multi-document AI assistant with advanced retrieval")

# ================= MULTI FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ================= CREATE / UPDATE DB =================
if uploaded_files:

    if st.button("Create / Update Knowledge Base"):

        with st.spinner("Processing documents..."):

            all_docs = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name

                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300
            )

            chunks = splitter.split_documents(all_docs)

            embeddings = MistralAIEmbeddings(model="mistral-embed")

            if os.path.exists("chroma_db"):
                vectorstore = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=embeddings
                )
                vectorstore.add_documents(chunks)
            else:
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

            vectorstore.persist()

        st.success("Knowledge base updated!")

# ================= LOAD DB =================
if os.path.exists("chroma_db"):

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    # ================= DOCUMENT FILTER =================
    all_sources = list(set(
    [meta.get("source", "Unknown") for meta in vectorstore.get()["metadatas"]]
))
    selected_doc = st.selectbox("Filter by Document (optional)", ["All"] + all_sources)

    # ================= PROMPT =================
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are an expert AI assistant.

Use provided context to answer accurately.

Rules:
- Combine multiple sources
- Be precise
- No hallucination
- If not found, say clearly
"""),
        ("human",
        """Chat History:
{chat_history}

Context:
{context}

Question:
{question}
""")
    ])

    st.divider()
    query = st.text_input("💬 Ask a question")

    if query:

        with st.spinner("Thinking..."):

            history_text = "\n".join(st.session_state.chat_history)

            # ================= QUERY REWRITE =================
            rewrite_prompt = f"""
Rewrite the query clearly.

Chat History:
{history_text}

Query:
{query}
"""
            rewritten_query = llm.invoke(rewrite_prompt).content

            # ================= RETRIEVE =================
            docs = retriever.invoke(rewritten_query)

            # ================= FILTER =================
            if selected_doc != "All":
                docs = [d for d in docs if d.metadata.get("source") == selected_doc]

            # ================= RERANK =================
            query_words = set(rewritten_query.lower().split())

            def score(doc):
                return len(query_words.intersection(set(doc.page_content.lower().split())))

            docs = sorted(docs, key=score, reverse=True)[:4]

            if not docs:
                st.warning("No relevant info found")
                st.stop()

            # ================= CONTEXT =================
            context = "\n\n".join([
                f"[{doc.metadata.get('source')}]\n{doc.page_content[:500]}"
                for doc in docs
            ])

            final_prompt = prompt.invoke({
                "context": context,
                "question": query,
                "chat_history": history_text
            })

            response = llm.invoke(final_prompt)

            # ================= STREAMING =================
            st.write("### 🤖 Answer")
            output_placeholder = st.empty()
            full_text = ""

            for word in response.content.split():
                full_text += word + " "
                output_placeholder.markdown(full_text)
                time.sleep(0.02)

            # ================= CONFIDENCE =================
            confidence = min(100, len(docs) * 20)
            st.metric("Confidence", f"{confidence}%")

            # ================= SAVE HISTORY =================
            st.session_state.chat_history.append(f"User: {query}")
            st.session_state.chat_history.append(f"AI: {response.content}")

            # ================= SOURCES =================
            st.write("### 📌 Sources")
            for doc in docs:
                st.write(f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
                st.write(doc.page_content[:150] + "...")

# ================= CHAT =================
st.divider()
st.subheader("💬 Chat History")

for msg in st.session_state.chat_history:
    st.write(msg)