# About
LangGraphBotDocIngestor is a ChatBot API designed to assist with answering questions about LangGraph and LangChain documentation. It leverages local embeddings, a retrieval-augmented generation (RAG) architecture, and a lightweight Llama-based local language model.


 # Features
- LangGraph & LangChain Focused – specifically tailored to provide accurate answers to these libraries.

- FastAPI Backend – lightweight, production-ready REST API.

- Local Embeddings with ChromaDB – no external vector database needed.

- TinyLlama LLM (Quantized) – fast local LLM execution via ctransformers.

- Session Memory – maintains conversational context for coherent interactions.

- Post-processing & Similarity Filtering – ensures meaningful and relevant responses.

- Sources Included – provides links to source documents used in answers.

# Directory Structure
LangGraphBotDocIngestor/
- chroma_db/              # Persistent storage for document embeddings
- loaders/
  - langchain_docs.py   # Ingest script for LangChain documentation
- models/                 # Quantized TinyLlama GGUF model
- .env                    # Environment variables
- ingest.py               # Script to build embedding database
- qa_api.py               # FastAPI app serving the Q&A endpoint
- memory.py               # In-memory session storage for conversations
- requirements.txt        # Development dependencies
- pinned-requirements.txt # Locked versions for production
- README.md               # Project documentation

# Getting Started
Clone the Repository

git clone https://github.com/ayorindeadunse/langgraph-project-bot-doc-ingestion-layer.git
cd LangGraphBotDocIngestor

# Set Up Environment Variables
Create a .env file in the root directory if required for specific secrets.

# Example .env:
USER_AGENT=LanggraphBotDocIngestor/1.x (your_email_here)
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here

- any other environmental variable that might be needed in the course of code update/refactoring

# Installing Dependencies
## For development:

bash
pip install -r requirements.txt

## For production environments

pip install -r pinned-requirements.txt

# Prepare the Vector Store 
run: python ingest.py

This will populate chroma_db/ with document embeddings.

# Running the API
bash
uvicorn qa_api:app --reload

The API  will be available at: http://127.0.0.1:8000 and the OpenAPI documentation at http://127.0.0.1:8000/docs 

# API Endpoints
GET /
Returns a status message confirming that the API is running.

POST /ask
Description: Submit a question and optionally provide a session_id for multi-turn conversations.

# Request Body
{
  "question": "What is LangGraph?"
}
# Response
{
  "response": "LangGraph extends LangChain to enable building applications as stateful graphs.",
  "session_id": "generated-or-provided-session-id"
}

# Tech Stack
- Python 3.11+

- FastAPI

- LangChain (Community, HuggingFace)

- CTransformers (for quantized TinyLlama)

- ChromaDB (via LangChain)

- SentenceTransformers (Embeddings)

- NumPy & Scikit-Learn (Similarity computation)

# Development Workflow
- Modify your code or models

- Test locally with uvicorn

- Commit and push changes

- Deploy with pinned dependencies for consistency

# License
MIT License © 2025 Ayorinde Adunse

# Front End Integration
The Client Web Appplication was build using Blazor on ASP.NET Core Web App Framework. The project is availabe here: https://github.com/ayorindeadunse/langgraph-docs-bot

# Contributing
Contributions welcome! Please submit issues or PRs.


