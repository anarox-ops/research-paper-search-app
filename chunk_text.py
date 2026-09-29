from langchain_text_splitters import MarkdownTextSplitter

# 1. Read your generated output.md file
with open("output.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

# 2. Initialize the Markdown splitter
splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)

# 3. Split the text into chunks
chunks = splitter.split_text(markdown_text)

print(f"Total chunks created: {len(chunks)}\n")

# 4. Print preview of each chunk
for i, chunk in enumerate(chunks):
    print(f"--- CHUNK {i+1} ---")
    print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    print("\n")