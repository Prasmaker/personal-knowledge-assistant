# 📚 Personal Knowledge Assistant using LangChain

A RAG-powered chatbot that lets you upload any PDF and have a conversation with it.
Built with LangChain, Groq (LLaMA 3.3), ChromaDB, and Streamlit.

## Features
- Upload any PDF document
- Ask questions in natural language
- Grounded answers with page citations
- Three answer styles: Default, Concise, Analyst

## Tech Stack
| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |

## How to run locally

1. Clone the repo
2. Install packages from requirements
   pip install -r requirements.txt
3. Create a free Groq API key and save it in a .env file
   GROQ_API_KEY=your_key_here
4. Run using
   streamlit run app.py

## Project Structure
- ingest.py       for PDF loading and chunking
- vectorstore.py  provides Embeddings and vector storage
- rag_chain.py    the RAG pipeline
- prompts.py      a separate file for Prompt engineering templates
- app.py          uses Streamlit UI as app front-end
