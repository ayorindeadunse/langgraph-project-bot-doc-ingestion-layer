import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from loaders.langchain_docs import load_documents
from tqdm import tqdm


# Load .env
load_dotenv()

def ingest_documents():
    print("Loading documents...")
    raw_docs = load_documents()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    print("Embedding and storing in Chroma...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_store"
    )

    vectorstore.persist()
    print(f"Ingested {len(chunks)} chunks into vectorstore.")

    if __name__ == "__main__":
        ingest_documents()

