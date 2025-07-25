# Embedding model to use for document encoding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# LLM Model
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf" 
MODEL_TYPE = "llama"

# Chroma vectorstore directory
CHROMA_PERSIST_DIR = "chroma_db"


# Text splitter parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
RETRIEVER_K = 3
MIN_CONTEXT_LENGTH = 30
SIMILARITY_THRESHOLD = 0.3


PROMPT_TEMPLATE = (
    "You are a helpful assistant specializing in LangGraph and LangChain documentation.\n"
    "Example Q&A:\n"
    "Q: What is LangChain?\n"
    "A: LangChain is an open-source framework for developing applications powered by language models.\n"
    "Q: What is LangGraph?\n"
    "A: LangGraph extends LangChain to enable building applications as stateful graphs.\n\n"
    "Now, using the following context:\n{context}\n\n"
    "Conversation history:\n{history}\n\n"
    "Q: {question}\nA:"
)