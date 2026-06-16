import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import BugReportRetriever
from src.generation import generate_bug_summary


# ============================================================
# RAG Pipeline Test
#
# Course connections:
# - Lecture15-FAISS: semantic retrieval with FAISS
# - Lecture15-Ollama: local LLM generation with Ollama
#
# Pipeline:
# User query -> FAISS retrieval -> retrieved context -> Ollama/Mistral response
# ============================================================


model_name = "mistral"

query = "application crashes when opening a file"

print("Building retriever...")

retriever = BugReportRetriever(
    dataset_path="data/bug_reports.csv",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)

retriever.build_index()

print("\nRetrieving similar bug reports...")

results = retriever.search_bug_reports(query, top_k=3)

context = retriever.format_results_for_llm(results, max_chars=400)

print("\n" + "=" * 100)
print("USER QUERY")
print("=" * 100)
print(query)

print("\n" + "=" * 100)
print("RETRIEVED CONTEXT")
print("=" * 100)
print(context)

print("\nGenerating answer with Ollama / Mistral...")

answer = generate_bug_summary(
    user_query=query,
    retrieved_context=context,
    model=model_name
)

print("\n" + "=" * 100)
print("GENERATED DEBUGGING REPORT")
print("=" * 100)
print(answer)