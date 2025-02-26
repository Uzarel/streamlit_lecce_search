import os
from typing import Optional

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
                mode="page",                        # Process the document page by page
                images_inner_format="html-img",     # Format for embedded images
                images_parser=TesseractBlobParser(langs=[OCR_LANG])  # Use Tesseract for OCR
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
        chunk_overlap=200  # Overlap to maintain context between chunks
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

# Define a synchronous function to find the most relevant document for a given query
def find_relevant_doc(query: str, similarity_threshold=0.7) -> Optional[str]:
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=1)
    if similar_docs:
        most_similar_doc, most_similar_score = similar_docs[0]
        print(f"Query: {query}")
        print(f"Most similar doc: {most_similar_doc.metadata.get('file_path', 'unknown')}")
        print(f"Most similar score: {most_similar_score}")
        if most_similar_score > similarity_threshold:
            return most_similar_doc.metadata.get("file_path")
    return None

# Streamlit UI
def main():
    st.title("App di ricerca documentale con IA")
    
    query = st.text_input("Inserire la chiave di ricerca:")

    if query:
        doc_path = find_relevant_doc(query)
        if doc_path:
            st.success(f"Trovato documento inerente: {doc_path}")
            if os.path.exists(doc_path):
                # Provide a download button for the PDF
                with open(doc_path, "rb") as file:
                    st.download_button(
                        label="Scarica PDF",
                        data=file,
                        file_name=os.path.basename(doc_path),
                        mime="application/pdf"
                    )
                # Display the PDF using the streamlit_pdf_viewer
                pdf_viewer(doc_path)
            else:
                st.error(f"Documento {doc_path} non trovato.")
        else:
            st.error("Nessun documento rilevante è stato trovato per la tua chiave di ricerca.")

if __name__ == "__main__":
    main()
