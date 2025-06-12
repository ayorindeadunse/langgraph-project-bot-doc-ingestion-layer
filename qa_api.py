import os
import uuid
import traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from memory import SessionMemory
from pydantic import BaseModel
from dotenv import load_dotenv

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
memory = SessionMemory()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()
retriever.search_kwargs = {"k": 10}

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, do_sample=False)
llm = HuggingFacePipeline(pipeline=pipe)

# Define prompt and LLMChain explicitly
prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=(
        "You are a helpful and friendly assistant. Use markdown links for references when possible.\n"
        "Answer step by step with reasoning when appropriate.\n"
        "Only use the provided context to answer. If the question is unrelated, say 'I don't know.'\n\n"
        "Conversation history:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "User's Question:\n{question}\n\n"
        "Answer:"
    )
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)


@app.get("/")
def read_root():
    return {"message": "QA API is running with self-hosted LLM!"}

@app.post("/ask")
async def ask_question(
    request: QueryRequest,
    session_id: str = Query(default=None)
):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        history_pairs = memory.get_session(session_id)
        history_text = "\n".join([f"User: {q}\nBot: {a}" for q, a in history_pairs])

        # Retrieve documents for the context
        docs = retriever.get_relevant_documents(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Run the LLM chain with explicitly all variables
        result = llm_chain.invoke({
            "history": history_text,
            "context": context,
            "question": request.question
        })

        if isinstance(result, dict):
            answer = result.get("text","").strip()
        else:
            answer = str(result).strip()

        # Save to session memory
        memory.add_message(session_id, request.question, answer)

        # Prepare links from metadata
        links = []
        seen = set()
        for doc in docs:
            meta = doc.metadata
            title = meta.get("title", "Untitled")
            source = meta.get("source", "")
            key = (title, source)
            if source and key not in seen:
                seen.add(key)
                links.append(f"- [{title}]({source})")

        full_response = f"{answer}\n\n**Sources:**\n" + "\n".join(links) if links else answer

        return {"response": full_response, "session_id": session_id}

    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})
