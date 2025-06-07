import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import pipeline,AutoModelForSeq2SeqLM,AutoTokenizer

#Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str

# Setup embeddings + vectorstore
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db",embedding_function=embedding)
retriever = vectorstore.as_retriever()

# self-hosted LLM setup
model_name = "google/flan-t5-small" # smaller model for local usage

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create Hugging Face pipeline for text2text generation
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False,
)

# Wrap pipeline into LangChain's HuggingFacePipeline LLM wrapper
llm = HuggingFacePipeline(pipeline=pipe)

# Setup the RetrievalQA chain with the self-hosted LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

@app.get("/")
def read_root():
    return {"message": "QA API is running with self-hosted LLM!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Manually retrieve first
       # retrieved_docs = retriever.invoke(request.question)
       # print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents")

        result = qa_chain.invoke({"query": request.question})

        # Deduplicate by unique combination of title + source url
        seen = set()
        unique_sources = []
        for doc in result["source_documents"]:
            meta = doc.metadata
            key = (meta.get("title"), meta.get("source"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(meta)

        return {
            "answer": result["result"],
            "sources": unique_sources,
        }
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb)