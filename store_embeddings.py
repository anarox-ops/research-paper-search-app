from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Read your Markdown file
with open("output.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

# 2. Split the text into chunks (same as before)
splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(markdown_text)

print(f"Total chunks to embed: {len(chunks)}")

# 3. Initialize a free, local embedding model from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the Vector Database and store the chunks
# This automatically embeds your chunks and saves them in a folder named "chroma_db"
vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Embeddings generated and saved successfully in './chroma_db'!")