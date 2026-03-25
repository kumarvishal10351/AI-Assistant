import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
import streamlit as st
load_dotenv()
import streamlit as st
import os

os.environ["MISTRAL_API_KEY"] = st.secrets["wEy0qYoYAOmAstKeLwdCOuArPXsSAVuZ"]



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Assitant", page_icon="🤖", layout="wide")

st.title("📚 AI Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = MistralAIEmbeddings(model="mistral-embed")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            vectorstore.persist()

        st.success("Vector database created!")



if os.path.exists("chroma_db"):

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )
    

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer.

Also consider the chat history for better understanding.

If the answer is not present, say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
        )
    ]
)

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        with st.spinner("Thinking..."):

            docs = retriever.invoke(query)

            if not docs:
                st.warning("No relevant information found.")
                st.stop()

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            chat_history = "\n".join(st.session_state.chat_history)

            final_prompt = prompt.invoke({
                "context": context,
                "question": query,
                "chat_history": chat_history
            })

            response = llm.invoke(final_prompt)

            # Save history
            st.session_state.chat_history.append(f"User: {query}")
            st.session_state.chat_history.append(f"AI: {response.content}")

            st.write("### 🤖 AI Answer")
            st.write(response.content)

            st.write("### 📌 Sources")
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", "N/A")
                st.write(f"Source {i+1} (Page {page}):")
                st.write(doc.page_content[:50] + "...")
st.divider()
st.subheader("💬 Chat History")

for msg in st.session_state.chat_history:
    st.write(msg)

       
        
           
        
        
