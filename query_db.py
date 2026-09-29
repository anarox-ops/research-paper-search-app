from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load embedding model and database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# 2. Ask any question about your PDF!
query = "What are the main deep-sea technologies discussed?"
print(f"Searching for: '{query}'\n")

# 3. Search database for top 3 matching chunks
results = vector_store.similarity_search(query, k=3)

# 4. Print results
for i, res in enumerate(results):
    print(f"--- RELEVANT CHUNK {i+1} ---")
    print(res.page_content[:500] + "...\n")