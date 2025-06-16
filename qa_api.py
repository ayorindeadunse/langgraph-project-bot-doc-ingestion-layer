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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever()
retriever.search_kwargs = {"k": 3}

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
tiny_llama = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="lllama",
    context_length=2048,
    gpu_layers=0 # So that any system without GPU can run it
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
)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

MIN_CONTEXT_LENGTH = 30
SIMILARITY_THRESHOLD = 0.3 

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
                "That topic appears unrelated, so I can't provide a reliable answer."
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

        # Semantic similarity filtering
        question_emb = embedding.embed_query(request.question)
        answer_emb = embedding.embed_query(answer)
        similarity = cosine_similarity(
            np.array(question_emb).reshape(1, -1),
            np.array(answer_emb).reshape(1, -1)
        )[0][0]

        if similarity < SIMILARITY_THRESHOLD:
            answer = (
                "I specialize in answering questions about LangGraph and LangChain documentation.\n"
                "That topic appears unrelated, so I can't provide a reliable answer."
            )
            
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
