__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import tempfile
import streamlit as st

import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Page Config
st.set_page_config(page_title="Multi-Paper Search Engine", layout="wide")

st.title("📚 Research Paper Search Engine")

# Initialize Embeddings & Vector DB
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = load_embeddings()
persist_directory = "./chroma_db"

import chromadb
from langchain_chroma import Chroma

# Initialize explicit persistent client to prevent Streamlit Cloud KeyError
persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)

vector_store = Chroma(
    client=client,
    embedding_function=embedding_model
)

# ---------------------------------------------------------
# SECTION 1: UPLOAD PAPERS (MAIN PAGE)
# ---------------------------------------------------------
st.header("1. Upload Papers")

uploaded_files = st.file_uploader(
    "Drag and drop PDF research papers here:",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Extract & Index Papers", type="primary"):
    if uploaded_files:
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        total_chunks = 0

        with st.spinner("Cleaning text & generating embeddings..."):
            for pdf_file in uploaded_files:
                # Save to temp file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name

                # Clean text extraction using PyMuPDF4LLM
                md_text = pymupdf4llm.to_markdown(tmp_path, header=False, footer=False)
                os.remove(tmp_path)

                # Chunking
                chunks = splitter.split_text(md_text)

                # Attach source metadata
                metadatas = [{"source": pdf_file.name} for _ in chunks]

                # Store in Chroma DB
                vector_store.add_texts(texts=chunks, metadatas=metadatas)
                total_chunks += len(chunks)

        st.success(f"Done! Successfully added {len(uploaded_files)} paper(s) ({total_chunks} total chunks).")
    else:
        st.error("Please select at least one PDF file first!")

st.divider()

# ---------------------------------------------------------
# SECTION 2: SEARCH PAPERS
# ---------------------------------------------------------
st.header("2. Search Papers")

query = st.text_input("Enter your question or keyword search:", placeholder="e.g. deep sea plants")

if query:
    results = vector_store.similarity_search(query, k=4)

    if results:
        st.subheader("Top Matches Found:")
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown PDF")
            with st.expander(f"Match #{idx} — Source: {source}", expanded=(idx == 1)):
                st.markdown(doc.page_content)
    else:
        st.info("No relevant content found in database.")