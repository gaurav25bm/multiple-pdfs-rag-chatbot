# 🤖 Multi-Document RAG Chatbot

**Status:** Live and Operational 🟢  
**🚀 [Click here to launch the Live Web App]
(https://multiple-pdfs-rag-chatbot-2dldzrrjv5eao5qexe5rtg.streamlit.app/)**

---

### 📊 Project Overview
This is a **Retrieval-Augmented Generation (RAG)** application designed to transform static PDF documents into interactive knowledge bases. By combining Large Language Models (LLMs) with vector search, the app allows users to query multiple complex documents and receive accurate, context-aware answers instantly.

### 🤖 AI & RAG Implementation
I developed a specialized pipeline to ensure the chatbot only answers based on the provided data (minimizing hallucinations):

* **Google Gemini 1.5 Pro:** Used as the core "brain" for natural language understanding and generating human-like responses.
* **Google Generative AI Embeddings:** Converts raw text into high-dimensional mathematical vectors to capture semantic meaning.
* **FAISS (Facebook AI Similarity Search):** A high-performance vector database used to index document chunks and perform lightning-fast similarity searches.
* **LangChain Orchestration:** Manages the "Conversational Retrieval Chain," connecting the PDF context to the LLM prompt.

### 📄 Data Pipeline Details
The system processes unstructured PDF data through a multi-stage workflow:
* **Text Extraction:** Utilizes `PyPDF2` to scrape raw text from uploaded files.
* **Recursive Chunking:** Splits text into 1,000-character segments with a 200-character overlap to maintain context across boundaries.
* **Vectorization:** Each chunk is embedded and stored in a local `faiss_index` for offline/reusable retrieval.

### 📈 Results & Performance
* **Accuracy:** Implemented "Context Injection," ensuring the AI cites only the uploaded PDFs.
* **Efficiency:** FAISS indexing allows for document retrieval in **milliseconds**, regardless of file size.
* **Scalability:** Optimized to handle multiple large PDFs simultaneously without losing conversation history.

### 🛠️ Built With
* **Python** (LangChain, PyPDF2, Python-Dotenv)
* **AI/ML** (Google Gemini API, FAISS, Google Embeddings)
* **UI/UX** (Streamlit)
* **Deployment** (Streamlit Community Cloud & GitHub)

---

### 📦 Quick Start (Local)
1. **Clone:** `git clone https://github.com/gaurav25bm/multiple-pdfs-rag-chatbot.git`
2. **Install:** `pip install -r requirements.txt`
3. **Key:** Add `GOOGLE_API_KEY` to a `.env` file.
4. **Run:** `streamlit run app.py`