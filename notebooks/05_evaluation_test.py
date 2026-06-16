import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import evaluate_from_judgments


# ============================================================
# Evaluation Test
#
# This script computes:
# - Precision@3
# - Precision@5
# - Mean Reciprocal Rank
# - NDCG@5
#
# based on manually assigned graded relevance judgments.
# ============================================================


results_df, summary_df = evaluate_from_judgments(
    judgment_path="data/evaluation_judgments.csv"
)

print("\n" + "=" * 100)
print("PER-QUERY EVALUATION RESULTS")
print("=" * 100)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("SUMMARY RESULTS")
print("=" * 100)
print(summary_df.to_string(index=False))

results_df.to_csv(
    "data/evaluation_results_per_query.csv",
    index=False,
    encoding="utf-8"
)

summary_df.to_csv(
    "data/evaluation_summary.csv",
    index=False,
    encoding="utf-8"
)

print("\nSaved:")
print("data/evaluation_results_per_query.csv")
print("data/evaluation_summary.csv")