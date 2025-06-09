import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from memory import SessionMemory

# Load environment variables
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
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# Self-hosted LLM setup
model_name = "google/flan-t5-base"  
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

# Custom prompt template to inject a friendly tone
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful and friendly assistant. Use markdown links for references when possible. "
        "Only use the provided context to answer the question. "
        "If the question is unrelated, respond with 'I don't know.'\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
)

# Setup RetrievalQA chain with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

@app.get("/")
def read_root():
    return {"message": "QA API is running with self-hosted LLM!"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Run RetrievalQA chain with the user's actual question
        result = qa_chain.invoke({"query": request.question})

        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        # Gracefully handle empty answers or unrelated questions
        if not answer or "i don't know" in answer.lower():
            return {
                "response": (
                    "Sorry, I couldn't find anything relevant for that. "
                    "Try asking something related to LangChain or LangGraph!"
                )
            }

        # Deduplicate source documents (by title + source URL)
        seen = set()
        links = []
        for doc in source_docs:
            meta = doc.metadata
            title = meta.get("title", "Untitled")
            source = meta.get("source", "")
            key = (title, source)
            if source and key not in seen:
                seen.add(key)
                links.append(f"- [{title}]({source})")

        # Append sources (if any) to the response
        if links:
            sources_md = "\n\n**Sources:**\n" + "\n".join(links)
            full_response = f"{answer}\n\n{sources_md}"
        else:
            full_response = answer

        return {"response": full_response}

    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb)
