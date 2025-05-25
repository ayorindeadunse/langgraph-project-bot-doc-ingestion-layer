from langchain.document_loaders import WebBaseLoader

# This file contains logic to load and clean docs from the web (LangChain/LangGraph)

# Add URLs  here from LangChain or LangGraph docs (limit to 20-50 pages as per project requirements)

URLS =  [
    "https://python.langchain.com/docs/introduction/"
]

def load_documents():
    loader = WebBaseLoader(URLS)
    return loader.load()