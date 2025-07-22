import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import config

# Load embedding model and vector store
embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
retriever = Chroma(persist_directory=config.CHROMA_PERSIST_DIR,embedding_function=embedding).as_retriever()
retriever.search_kwargs = {"k": config.RETRIEVER_K}

# Load evaluation test set
test_file = os.path.join("tests", "eval_data.json")
with open(test_file) as f:
    test_data = json.load(f)

# Run retrieval and compute Recall@k
hits = 0
for sample in test_data:
    query = sample["question"]
    target_snippet = sample["relevant_doc_contains"].lower()

    results = retriever.get_relevant_documents(query)
    found = any(target_snippet in doc.page_content.lower() for doc in results)

    print(f"Q: {query} -> {'Found' if found else 'Missed'}")
    hits += int(found)

recall = hits / len(test_data)
print(f"\n Recall@{config.RETRIEVER_K}: {recall:.2f}")