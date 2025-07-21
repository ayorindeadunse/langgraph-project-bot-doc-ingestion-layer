import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loaders.langchain_docs import load_documents
from tqdm import tqdm
import config

# Load .env variables
load_dotenv()

def ingest_documents():
    
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        print("Cleaning up existing Chroma vectorstore...")
        shutil.rmtree(config.CHROMA_PERSIST_DIR)

    print("Loading documents...")
    raw_docs = load_documents()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)

    print("Embedding and storing in Chroma (locally)...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR
    )

    vectorstore.persist()
    print(f"Ingested {len(chunks)} chunks into vectorstore.")

if __name__ == "__main__":
    ingest_documents()
