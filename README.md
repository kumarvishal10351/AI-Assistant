# 🧠 AI Knowledge Assistant (RAG-Based)

An intelligent AI-powered assistant that allows users to upload PDF documents and ask questions based on their content. The system uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers with source references.

---

## 🚀 Features

- 📄 Upload and process PDF documents
- 🔍 Context-aware question answering
- 🧠 RAG pipeline using embeddings + vector search
- 📌 Source citations with page references
- 💬 Conversational memory (chat history)
- ⚡ Fast retrieval using ChromaDB
- 🌐 Deployable with Streamlit Cloud

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Mistral AI (`mistral-small`)
- **Embeddings:** Mistral Embeddings (`mistral-embed`)
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Language:** Python

---

## 🧠 How It Works (Architecture)

1. User uploads a PDF
2. Document is split into smaller chunks
3. Chunks are converted into embeddings
4. Stored in a vector database (ChromaDB)
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates answer using context

---

## 📂 Project Structure

```
AI-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/kumarvishal10351/AI-Assistant.git
cd AI-Assistant
```

---

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Add API Key

Create a `.env` file:

```
MISTRAL_API_KEY=your_api_key_here
```

---

### 5. Run the app

```
streamlit run app.py
```

---

## ☁️ Deployment

The app can be deployed using **Streamlit Cloud**.

- Push code to GitHub
- Connect repo to Streamlit Cloud
- Add API key in Secrets

---

## 🧪 Example Use Cases

- 📚 Study assistant for books/notes
- 📄 Research paper analysis
- 🏢 Internal company knowledge assistant
- 🧾 Document-based Q&A system

---

## ⚠️ Limitations

- Answers depend on document quality
- May fail if context is not retrieved correctly
- Requires API key for Mistral

---

## 🔥 Future Improvements

- 🌐 Web + document hybrid search
- 🎤 Voice input/output
- 📊 Confidence scoring
- 🧠 Advanced query rewriting
- 🎨 Improved UI (React frontend)

---

## 👨‍💻 Author

**Vishal Kumar**
GitHub: https://github.com/kumarvishal10351

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
