import pymupdf4llm

# Extract text as clean Markdown, stripping headers and footers
# Replace "your_document.pdf" with the exact name of your PDF file
md_text = pymupdf4llm.to_markdown(
    "Deep-seaorganismsresearch.pdf", header=False, footer=False
)

# Save the extracted text to a Markdown file
with open("output.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("Extraction complete! Check output.md")