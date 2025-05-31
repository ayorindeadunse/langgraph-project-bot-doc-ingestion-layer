from langchain_community.document_loaders import WebBaseLoader
from  dotenv import load_dotenv
import os

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "LanggraphBotDocIngestor/1.0 (adunseayorinde@gmail.com)")
HEADERS = {"User-Agent": USER_AGENT}

# This file contains logic to load and clean docs from the web (LangChain/LangGraph)

# Add URLs  here from LangChain or LangGraph docs (limit to 20-50 pages as per project requirements)

URLS =  [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/how_to/",
    "https://python.langchain.com/docs/concepts/",
    "https://docs.smith.langchain.com/?_gl=1*p2cd58*_ga*MTU0ODA4OTM3OC4xNzQ4MDg4MzI1*_ga_47WX3HKKY2*czE3NDgxOTkwNDkkbzQkZzEkdDE3NDgxOTkxNTEkajAkbDAkaDA.",
    "https://langchain-ai.github.io/langgraph/?_gl=1*qr6mrp*_gcl_au*MTY3ODE3NzY0NS4xNzQ4MTk5MTg5*_ga*MTU0ODA4OTM3OC4xNzQ4MDg4MzI1*_ga_47WX3HKKY2*czE3NDgxOTkwNDkkbzQkZzEkdDE3NDgxOTkxOTYkajAkbDAkaDA.",
    "https://python.langchain.com/docs/versions/v0_3/",
    "https://python.langchain.com/docs/versions/v0_2/overview/",
    "https://python.langchain.com/docs/versions/v0_2/",
    "https://python.langchain.com/docs/versions/v0_2/migrating_astream_events/",
    "https://python.langchain.com/docs/versions/v0_2/deprecations/",
    "https://python.langchain.com/docs/how_to/pydantic_compatibility/",
    "https://python.langchain.com/docs/versions/migrating_chains/",
    "https://python.langchain.com/docs/versions/migrating_memory/",
    "https://python.langchain.com/docs/versions/release_policy/",
    "https://python.langchain.com/docs/security/",
    "https://python.langchain.com/docs/integrations/providers/",
    "https://python.langchain.com/api_reference/",
    "https://python.langchain.com/docs/contributing/",
    "https://python.langchain.com/docs/people/",
    "https://python.langchain.com/docs/troubleshooting/errors/",
    "https://python.langchain.com/docs/integrations/providers/anthropic/",
    "https://python.langchain.com/docs/integrations/providers/aws/",
    "https://python.langchain.com/docs/integrations/providers/google/",
    "https://python.langchain.com/docs/integrations/providers/huggingface/",
    "https://python.langchain.com/docs/integrations/providers/microsoft/",
    "https://python.langchain.com/docs/integrations/providers/openai/",
    "https://python.langchain.com/docs/integrations/chat/",
    "https://python.langchain.com/docs/integrations/retrievers/",
    "https://python.langchain.com/docs/integrations/tools/",
    "https://python.langchain.com/docs/integrations/document_loaders/",
    "https://python.langchain.com/docs/integrations/vectorstores/",
    "https://python.langchain.com/docs/integrations/text_embedding/",
    "https://python.langchain.com/docs/integrations/llms/",
    "https://python.langchain.com/docs/integrations/stores/",
    "https://python.langchain.com/docs/integrations/document_transformers/",
    "https://python.langchain.com/docs/integrations/llm_caching/",
    "https://python.langchain.com/docs/integrations/graphs/",
    "https://python.langchain.com/docs/integrations/memory/",
    "https://python.langchain.com/docs/integrations/callbacks/",
    "https://python.langchain.com/docs/integrations/chat_loaders/",
    "https://python.langchain.com/docs/integrations/adapters/",
    "https://python.langchain.com/docs/integrations/providers/abso/",
    "https://python.langchain.com/docs/integrations/providers/acreom/",
    "https://python.langchain.com/docs/integrations/providers/activeloop_deeplake/",
    "https://python.langchain.com/docs/integrations/providers/ads4gpts/",
    "https://python.langchain.com/docs/integrations/providers/aerospike/",
    "https://python.langchain.com/docs/integrations/providers/agentql/",
    "https://python.langchain.com/docs/integrations/providers/ai21/",
    "https://python.langchain.com/docs/integrations/providers/aim_tracking/",
    "https://python.langchain.com/docs/integrations/providers/ainetwork/"
]


def load_documents():
    loader = WebBaseLoader(web_paths=URLS, header_template=HEADERS)
    return loader.load()