import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate


# ----------------------------------------
# Environment Setup
# ----------------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

genai.configure(api_key=api_key)


# ----------------------------------------
# Cache embeddings (loads once)
# ----------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )


# ----------------------------------------
# Read PDFs
# ----------------------------------------
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        text += f"\n\nDocument: {pdf.name}\n\n"

        reader = PdfReader(pdf)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# ----------------------------------------
# Chunking (faster + better)
# ----------------------------------------
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ----------------------------------------
# Create FAISS Vector Store
# ----------------------------------------
def get_vector_store(text_chunks):

    try:

        if not text_chunks:
            st.error("No chunks found.")
            return

        st.write(
            f"Creating embeddings for {len(text_chunks)} chunks..."
        )

        embeddings = load_embeddings()

        st.write("Generating vector store...")

        vector_store = FAISS.from_texts(
            text_chunks,
            embedding=embeddings
        )

        st.write("Saving FAISS index...")

        vector_store.save_local("faiss_index")

        st.success(
            "✅ FAISS index created successfully!"
        )

    except Exception as e:
        st.error(
            f"FAISS creation failed: {e}"
        )


# ----------------------------------------
# Gemini QA Chain
# ----------------------------------------
def get_conversational_chain():

    prompt_template = """
Answer using the context.

If answer is not available in context say:
Answer is not available in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# ----------------------------------------
# Ask Questions
# ----------------------------------------
def user_input(user_question):

    try:

        if not os.path.exists("faiss_index"):
            st.error(
                "Upload and process PDFs first."
            )
            return

        embeddings = load_embeddings()

        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(
            user_question,
            k=4
        )

        chain = get_conversational_chain()

        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )

        st.write(
            "Reply:",
            response["output_text"]
        )

    except Exception as e:
        st.error(
            f"Error during question answering: {e}"
        )


# ----------------------------------------
# Main App
# ----------------------------------------
def main():

    st.set_page_config(
        page_title="Chat PDF"
    )

    st.header(
        "Chat with Multiple PDFs using Gemini 💁"
    )

    user_question = st.text_input(
        "Ask a Question from PDFs"
    )

    if user_question:
        user_input(user_question)


    with st.sidebar:

        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDFs and click Submit & Process",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning(
                    "Please upload PDFs."
                )
                st.stop()

            with st.spinner("Processing..."):

                raw_text = get_pdf_text(
                    pdf_docs
                )

                if not raw_text.strip():
                    st.error(
                        "Could not read PDFs."
                    )
                    st.stop()

                text_chunks = get_text_chunks(
                    raw_text
                )

                get_vector_store(
                    text_chunks
                )

                st.success("Done")


if __name__ == "__main__":
    main()