import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import BugReportRetriever


# ============================================================
# Evaluation Candidate Generator
#
# Course connection:
# Information Retrieval Evaluation
#
# This script retrieves top-5 bug reports for a set of
# natural language queries. The retrieved results will be
# manually judged with graded relevance:
#
# 2 = highly relevant
# 1 = partially relevant
# 0 = not relevant
# ============================================================


evaluation_queries = [
    "opening invalid scene file crashes the frontend",
    "window watcher callback error",
    "crash when opening location picker",
    "filter is broken and does not return links",
    "buttons do not work in listing view",
    "application freezes and work is lost",
    "login fails with password sync disabled",
    "error occurs when clicking save button",
    "app details do not load on iOS",
    "parsing failed when copying from iCloud notes"
]


print("Building retriever for evaluation...")

retriever = BugReportRetriever(
    dataset_path="data/bug_reports.csv",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)

retriever.build_index()

rows = []

for query in evaluation_queries:
    print(f"Retrieving top-5 results for query: {query}")

    results = retriever.search_bug_reports(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        rows.append({
            "query": query,
            "rank": rank,
            "bug_id": result.get("bug_id"),
            "title": result.get("title", ""),
            "faiss_distance_score": result.get("score"),
            "label": result.get("label", "N/A"),
            "description": result.get("description", "")[:900],
            "relevance": ""
        })


candidate_df = pd.DataFrame(rows)

candidate_df.to_csv(
    "data/evaluation_candidates.csv",
    index=False,
    encoding="utf-8"
)

print("\nSaved:")
print("data/evaluation_candidates.csv")
print("\nOpen this file and manually fill the 'relevance' column with 0, 1, or 2.")