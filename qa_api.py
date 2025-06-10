import os
import traceback
import uuid
from fastapi import FastAPI, HTTPException, Query
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

# Initialize session memory
memory =  SessionMemory()

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
    input_variables=["history", "context", "question"],
    template=(
        "You are a helpful and friendly assistant. Use markdown links for references when possible.\n"
        "Answer step by step with reasoning when appropriate.\n"
        "Only use the provided context to answer. If the question is unrelated, say 'I don't know.'\n\n"
        "Conversation history:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "User's Question:n{question}\n\n"
        "Answer:"
    )
)


@app.get("/")
def read_root():
    return {"message": "QA API is running with self-hosted LLM!"}

@app.post("/ask")
async def ask_question(
    request: QueryRequest,
    session_id: str = Query(default=None)
    ):
    try:
        # Generate a new session ID if one wasn't provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Get conversation history for this session
        history_pairs = memory.get_session(session_id)
        history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history_pairs])

        # Run RetrievalQA with user's question to get the context
        result = qa_chain_with_history(request.question, history_text)

        answer = result.get("result", "").strip()
        source_docs = result.get("source_documents", [])

        # Add the Q&A to the memory
        memory.add_message(session_id, request.question, answer)

        # Gracefully handle empty answers or unrelated questions
        if not answer or "i don't know" in answer.lower(): # refactor much later to be able to answer questions with multiple use cases
            return {
                "response": (
                    "Sorry, I couldn't find anything relevant for that. "
                    "Try asking something related to LangChain or LangGraph!"
                ),
                "session_id": session_id
            }

        # Append sources (if any)
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

        if links:
            sources_md = "\n\n**Sources:**\n" + "\n".join(links)
            full_response = f"{answer}\n\n{sources_md}"
        else:
            full_response = answer

        return {"response": full_response, "session_id": session_id}
    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail=tb)
    
def qa_chain_with_history(question: str, history: str):
    # Create new RetrievalQA on each request with the  updated prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=prompt_template.template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return qa_chain.invoke({
        "context": "", # Let retriever fill this in
        "question": question,
        "history": history
    })
