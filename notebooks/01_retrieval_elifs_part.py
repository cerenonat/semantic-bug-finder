import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import BugReportRetriever


# ============================================================
# Elif's Retrieval Test Script
# Semantic Bug Finder
# ============================================================

retriever = BugReportRetriever(
    dataset_path="data/bug_reports.csv",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)


df = retriever.load_dataset()

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


retriever.build_index()

# Test queries
test_queries = [
    "application crashes when opening a file",
    "memory leak after running the program for a long time",
    "browser freezes when loading a page",
    "error occurs during installation",
    "program does not respond after clicking button"
]

print("\n" + "=" * 100)
print("SEARCH TESTS")
print("=" * 100)

for query in test_queries:
    print("\nQUERY:", query)

    results = retriever.search_bug_reports(query, top_k=3)

    for r in results:
        print("\nBUG ID:", r["bug_id"])
        print("FAISS DISTANCE SCORE:", round(r["score"], 4))
        print("SEVERITY:", r.get("severity", "N/A"))
        print("DESCRIPTION:", r["description"][:400])
        print("-" * 80)


# Example output for Ceren's Ollama module
example_query = "application crashes when opening a file"
example_results = retriever.search_bug_reports(example_query, top_k=3)
context_for_llm = retriever.format_results_for_llm(example_results)

print("\n" + "=" * 100)
print("CONTEXT FOR OLLAMA / LLM")
print("=" * 100)
print(context_for_llm)


# Save processed dataset
retriever.save_processed_dataset("data/processed_bug_reports.csv")