from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def load_pdf(file_path: str):
    """
    Loads a PDF file and returns a list of Document objects.
    Each Document = one page of the PDF, with the text and metadata.
    """
    print(f"Loading PDF: {file_path}")
    
    loader = PyPDFLoader(file_path)  # point LangChain at the PDF
    pages = loader.load()            # reads every page into a list

    print(f"Total pages loaded: {len(pages)}")
    return pages

def split_into_chunks(pages: list):
    """
    Takes a list of pages and splits them into smaller overlapping chunks.
    Returns a new list of Document objects — one per chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk is max 500 characters
        chunk_overlap=50,     # consecutive chunks share 50 characters
        separators=["\n\n", "\n", ".", " "]  # try splitting here, in order
    )

    chunks = splitter.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# testing the loader
if __name__ == "__main__":
    pages = load_pdf(r"C:\Users\sw\pka\docs\wharton_casebook.pdf")  #sample pdf
    chunks = split_into_chunks(pages)

    # Inspect the first 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Length: {len(chunk.page_content)} chars")
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")