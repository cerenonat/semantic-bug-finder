import math
import pandas as pd


# ============================================================
# Evaluation Module for Semantic Bug Finder
#
# Course connections:
# - Information Retrieval Evaluation
# - Precision@k
# - Mean Reciprocal Rank
# - DCG / NDCG
#
# Relevance scores:
# 2 = highly relevant
# 1 = partially relevant
# 0 = not relevant
# ============================================================


def precision_at_k(relevance_scores, k):
    top_k = relevance_scores[:k]

    if len(top_k) == 0:
        return 0.0

    relevant_count = sum(1 for score in top_k if score > 0)

    return relevant_count / k


def reciprocal_rank(relevance_scores):
    for rank, score in enumerate(relevance_scores, start=1):
        if score > 0:
            return 1 / rank

    return 0.0


def dcg_at_k(relevance_scores, k):
    top_k = relevance_scores[:k]

    dcg = 0.0

    for rank, relevance in enumerate(top_k, start=1):
        dcg += relevance / math.log2(rank + 1)

    return dcg


def ndcg_at_k(relevance_scores, k):
    actual_dcg = dcg_at_k(relevance_scores, k)

    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_from_judgments(judgment_path="data/evaluation_judgments.csv"):
    df = pd.read_csv(judgment_path)

    if "relevance" not in df.columns:
        raise ValueError("The judgment file must contain a 'relevance' column.")

    df = df.dropna(subset=["relevance"])
    df["relevance"] = df["relevance"].astype(int)

    per_query_results = []

    for query, group in df.groupby("query"):
        group = group.sort_values("rank")

        relevance_scores = group["relevance"].tolist()

        p_at_3 = precision_at_k(relevance_scores, 3)
        p_at_5 = precision_at_k(relevance_scores, 5)
        rr = reciprocal_rank(relevance_scores)
        ndcg_5 = ndcg_at_k(relevance_scores, 5)

        per_query_results.append({
            "query": query,
            "Precision@3": round(p_at_3, 4),
            "Precision@5": round(p_at_5, 4),
            "Reciprocal Rank": round(rr, 4),
            "NDCG@5": round(ndcg_5, 4)
        })

    results_df = pd.DataFrame(per_query_results)

    summary = {
        "Mean Precision@3": round(results_df["Precision@3"].mean(), 4),
        "Mean Precision@5": round(results_df["Precision@5"].mean(), 4),
        "MRR": round(results_df["Reciprocal Rank"].mean(), 4),
        "Mean NDCG@5": round(results_df["NDCG@5"].mean(), 4)
    }

    summary_df = pd.DataFrame([summary])

    return results_df, summary_df