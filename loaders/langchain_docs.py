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
    "https://python.langchain.com/docs/integrations/vectorstores/",
    "https://python.langchain.com/docs/integrations/text_embedding/",
    "https://python.langchain.com/docs/integrations/llms/",
    "https://python.langchain.com/docs/integrations/stores/",
    "https://python.langchain.com/docs/integrations/document_transformers/"
]

def load_documents():
    loader = WebBaseLoader(web_paths=URLS, header_template=HEADERS)
    return loader.load()