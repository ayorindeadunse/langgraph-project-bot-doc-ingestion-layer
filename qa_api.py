import os
import uuid
import traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ctransformers import AutoModelForCausalLM
from langchain.llms.base import LLM
from memory import SessionMemory
from pydantic import BaseModel, PrivateAttr
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

memory = SessionMemory()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()
retriever.search_kwargs = {"k": 3}

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
tiny_llama = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="lllama",
    context_length=2048,
    gpu_layers=0
)

class CTransformersLLM(LLM):
    _model: any = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model

    def _call(self, prompt, stop=None, run_manager=None):
        return self._model(prompt)

    @property
    def _llm_type(self):
        return "ctransformers"

llm = CTransformersLLM(tiny_llama)

# Updated prompt - less likely to be echoed in output
prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=(
       "You are a helpful  assistant specialized in  LangGraph  and LangChain documentation. \n"
        "Respond to the  user's question clearly and  directly using only the context provided. \n"
        "Do  NOT  repeat the  user's question.If the answer is not  in the context, say \"I don't know\".\n"
        "Respond in a natural, helpful tone. Use markdown formatting (e.g., bullet points, links) when useful.  \n\n"
        "Conversation history:\n{history}\n\n"
        "Documentation Context:\n{context}\n\n"
        "User's Question:\n{question}\n\n"
        "**Your Answer:**"
    )
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

MIN_CONTEXT_LENGTH = 30

@app.get("/")
def read_root():
    return {"message": "QA API is running"}

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

        docs = retriever.get_relevant_documents(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])

        if not context.strip() or len(context.strip()) < MIN_CONTEXT_LENGTH:
            fallback_answer = (
                "I specialize in answering questions about LangGraph and LangChain documentation. \n"
                "That topic appears unrelated,  so I can't provide  a reliable answer."
            )
            return  {"response": fallback_answer, "session_id": session_id}

        result = llm_chain.invoke({
            "history": history_text,
            "context": context,
            "question": request.question
        })
        raw_answer = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

        # Post-processing to remove prompt echoes
        def clean_output(text: str) -> str:
            keywords = ["Helpful information:", "Previous conversation:", "Context:", "User's Question:", "Question:", "Answer:", "**Your Answer:**"]
            for key in keywords:
                if key in text:
                    text = text.split(key)[-1].strip()
            text = text.strip(" \"'\n\t")

            # Attempt to remove echoes of the question as  the answer
            if text.endswith("?"):
                lines = text.splitlines()
                if len(lines) == 1:
                    text = "" # very likely just an echoed question
                elif lines[-1].strip() == lines[-2].strip():
                    text = "\n".join(lines[:-1]).strip()
            return text

        answer = clean_output(raw_answer)

        memory.add_message(session_id, request.question, answer)

        links = []
        seen = set()
        for doc in docs:
            meta = doc.metadata
            title = meta.get("title", "Untitled")
            source = meta.get("source", "")

            # Guard: Ensure source is a valid URL
            if source and (source.startswith("http://") or source.startswith("https://")):
                key = (title, source)
                if key not in seen:
                    seen.add(key)
                    links.append(f"- [{title}]({source})")

        # Optional:fallback when no valid sources found
        if not links and docs:
            links.append("*No valid source URLs provided.*")

        full_response = f"{answer}\n\n**Sources:**\n" + "\n".join(links) if links else answer
        return {"response": full_response, "session_id": session_id}

    except Exception as e:
        tb = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": tb})
