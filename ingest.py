import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loaders.langchain_docs import load_documents
from tqdm import tqdm


# Load .env
load_dotenv()

def ingest_documents():
    chroma_dir  = "./chroma_db"
    if os.path.exists(chroma_dir):    
        print("🧹 Cleaning up existing Chroma vectorstore...")
        shutil.rmtree(chroma_dir)
   
    print("📄 Loading documents...")
    raw_docs = load_documents()

    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    print("🧠 Embedding and storing in Chroma (locally)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir
    )

    vectorstore.persist()
    print(f"Ingested {len(chunks)} chunks into vectorstore.")
   
if __name__ == "__main__":
        ingest_documents()

