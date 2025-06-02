import os
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS for local frontend development

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema
class QueryRequest(BaseModel):
    question:str

#Load vector store
persist_directory =  "chroma_db"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Setup retriever
retriever = vectorstore.as_retriever()

# Initialize Hugging Face LLM (no OpenAI required)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.1,
    max_new_tokens=512
)

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever=retriever,
    return_source_documents=True,
)

@app.get("/")
def read_root():
    return {"message":"QA API is running!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        result = qa_chain.invoke({"query": request.question})
        return {
            "answer":result["result"],
            "sources":[doc.metadata for doc in result["source_documents"]],
        }
    except Exception  as e:
        raise HTTPException(status_code=500, detail=str(e))