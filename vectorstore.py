from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#saving embeddings on disk and specifying embedding model
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"



def get_embedding_model():
    """
    Loads the embedding model.
    First run downloads it (~90MB). After that it's cached locally.
    """
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}   # use "cuda" if you have a GPU
    )
    print("Embedding model loaded.")
    return embeddings


def create_vectorstore(chunks: list):
    """
    Takes chunks from Phase 2, embeds each one,
    and stores them in ChromaDB on disk.
    """
    print(f"Embedding {len(chunks)} chunks and storing in ChromaDB...")

    embeddings = get_embedding_model()

#gets the embedding model, takes chunks, feeds to the model, generates and saves the vectors in chromadb

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR       
    )

    print(f"Done! Vectorstore saved to '{CHROMA_DIR}/'")
    return vectorstore

def load_vectorstore():
    """
    Loads an already-existing ChromaDB from disk.
    Use this after the first run — no need to re-embed every time.
    """
    print("Loading existing vectorstore from disk...")
    embeddings = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print("Vectorstore loaded.")
    return vectorstore

#testing with sample pdf
if __name__ == "__main__":
    # Import from ingest.py 
    from ingest import load_pdf, split_into_chunks

    # Load and chunk the PDF
    pages = load_pdf(r"C:\Users\sw\pka\docs\wharton_casebook.pdf")
    chunks = split_into_chunks(pages)

    # Embed and store
    vectorstore = create_vectorstore(chunks)

    # --- test semantic search! ---
    print("\n--- Testing Semantic Search ---")
    query = "What issues are discussed in Case US Shoe Manufacturing?"

    # embedding the query and return 3 closest chunks
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nResult {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:300])

