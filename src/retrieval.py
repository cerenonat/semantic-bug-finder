import re
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class BugReportRetriever:
    def __init__(
        self,
        dataset_path="data/bug_reports.csv",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    ):
        self.dataset_path = dataset_path
        self.embedding_model_name = embedding_model_name

        self.df = None
        self.model = None
        self.index = None
        self.embeddings = None

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"`", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_path)

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["body"] = self.df["body"].fillna("").astype(str)

        self.df["text"] = (
            self.df["title"].apply(self.clean_text)
            + ". "
            + self.df["body"].apply(self.clean_text)
        )

        self.df = self.df[self.df["text"].str.len() > 20].reset_index(drop=True)
        self.df["bug_id"] = range(len(self.df))

        return self.df

    def build_index(self):
        if self.df is None:
            self.load_dataset()

        print("Loading embedding model...")
        self.model = SentenceTransformer(self.embedding_model_name)

        print("Encoding bug reports...")
        self.embeddings = self.model.encode(
            self.df["text"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False
        ).astype("float32")

        dimension = self.embeddings.shape[1]

        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(dimension)

        print("Adding bug reports to FAISS index...")
        self.index.add(self.embeddings)

        print(f"FAISS index created with {self.index.ntotal} documents.")

    def search_bug_reports(self, query, top_k=3):
        if self.index is None:
            self.build_index()

        query = self.clean_text(query)

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            row = self.df.iloc[int(idx)]

            results.append({
                "bug_id": int(row.get("bug_id", idx)),
                "title": row.get("title", "N/A"),
                "label": row.get("label", "N/A"),
                "severity": row.get("Severity", "N/A"),
                "description": row.get("text", ""),
                "score": float(distance)
            })

        return results

    def format_results_for_llm(self, results, max_chars=800):
        context = ""

        for i, r in enumerate(results, start=1):
            short_description = r["description"][:max_chars]

            context += f"""
Retrieved Bug Report {i}
Bug ID: {r['bug_id']}
Title: {r.get('title', 'N/A')}
FAISS Distance Score: {r['score']:.4f}
Label: {r.get('label', 'N/A')}
Description: {short_description}
"""

        return context

    def save_processed_dataset(self, output_path="data/processed_bug_reports.csv"):
        if self.df is None:
            self.load_dataset()

        self.df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to {output_path}")