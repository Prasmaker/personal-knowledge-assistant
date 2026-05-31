from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from vectorstore import load_vectorstore
from prompts import get_rag_prompt

load_dotenv()


def get_llm():
    """
    Creates and returns the Groq LLM client.
    This is the model that will generate answers.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,      # 0 = focused/deterministic, 1 = creative/random
        max_tokens=1024       # max length of the answer
    )
    return llm


def build_rag_chain(vectorstore, prompt_mode: str = "default"):
    """
    Connects the retriever (ChromaDB) + prompt template + LLM
    into a single chain that handles everything automatically.
    """

    # --- Step 1: Turn vectorstore into a retriever ---
    # A retriever is just a wrapper that calls similarity_search() for us
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}    # fetch top 3 relevant chunks
    )

    #--- Step 2: Get prompt from the prompts file
    prompt = get_rag_prompt(mode=prompt_mode)

    # --- Step 3: Build the chain ---
    
    # user question → retriever → prompt → LLM → answer
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain


def ask(chain, question: str):
    """
    Asks a question through the RAG chain and prints
    both the answer and its sources.
    """
    print(f"\nQuestion: {question}")
    print("-" * 50)

    result = chain.invoke({"input": question})

    # LLM replies
    print(f"Answer:\n{result['answer']}")

    # which chunks were used to generate the answer
    print("\n--- Sources Used ---")
    for i, doc in enumerate(result['context']):
        page = doc.metadata.get('page', '?')
        print(f"  Source {i+1}: Page {page}")
        print(f"  Preview: {doc.page_content[:150]}...")

    return result


# --- test using a sample question ---
if __name__ == "__main__":
    # Loading the vectorstore from earlier  
    vectorstore = load_vectorstore()

    question = "What is the total addressable market for snacking almonds in US?"

    # Testing for all the prompt modes 
    for mode in ["default", "concise", "analyst"]:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")
        chain = build_rag_chain(vectorstore, prompt_mode=mode)
        ask(chain, question)