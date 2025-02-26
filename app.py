import os
from typing import List, Optional

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import TesseractBlobParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Set the OCR language and the folder containing your PDFs
OCR_LANG = "ita"
DOCS_FOLDER = "docs"

# Cache the loading of documents and creation of the vector store to speed up subsequent runs
@st.cache_resource
def load_vector_store():
    all_docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOCS_FOLDER, filename)
            print(f"Loading {pdf_path} ...")
            loader = PyMuPDFLoader(
                pdf_path,
                mode="page",
                images_inner_format="html-img",
                images_parser=TesseractBlobParser(langs=[OCR_LANG])
            )
            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_path}, error: {e}")
                continue

    print(f"Total pages loaded: {len(all_docs)}")

    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(split_docs)}")

    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings()
    
    # Create the vector store index from document chunks
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Load the vector store once at startup
vector_store = load_vector_store()

# Define a function to find the three most relevant documents for a given query
def find_relevant_docs(query: str, similarity_threshold=0.7) -> List[str]:
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=3)
    relevant_docs = set()
    
    for doc, score in similar_docs:
        print(f"Query: {query}")
        print(f"Document: {doc.metadata.get('file_path', 'unknown')}, Score: {score}")
        if score > similarity_threshold:
            relevant_docs.add(doc.metadata.get("file_path"))
    
    return relevant_docs

# Function to list available documents
def list_available_documents():
    """Returns a list of PDF filenames in the docs folder."""
    return [filename for filename in os.listdir(DOCS_FOLDER) if filename.lower().endswith(".pdf")]

# Streamlit UI
def main():
    st.set_page_config(layout="wide")

    st.title("App di ricerca documentale con IA")
    
    # Sidebar: Display available documents
    st.sidebar.header("📂 Documenti disponibili")
    available_docs = list_available_documents()
    if available_docs:
        for doc in available_docs:
            st.sidebar.write(f"📄 {doc}")
    else:
        st.sidebar.write("Nessun documento disponibile.")

    # Search input
    query = st.text_input("Inserire la chiave di ricerca:")

    if query:
        doc_paths = find_relevant_docs(query)
        if doc_paths:
            num_docs = len(doc_paths)
            cols = st.columns(num_docs)
            
            for i, doc_path in enumerate(doc_paths):
                with cols[i]:
                    if doc_path and os.path.exists(doc_path):
                        st.write(f"**Documento {i+1}:** {os.path.basename(doc_path)}")
                        with open(doc_path, "rb") as file:
                            st.download_button(
                                label=f"Scarica {os.path.basename(doc_path)}",
                                data=file,
                                file_name=os.path.basename(doc_path),
                                mime="application/pdf"
                            )
                        pdf_viewer(doc_path)
                    else:
                        st.error(f"Documento {doc_path} non trovato.")
        else:
            st.error("Nessun documento rilevante è stato trovato per la tua chiave di ricerca.")

if __name__ == "__main__":
    main()
