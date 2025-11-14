# experiments/t2_retrieval/evaluation.py
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from generate_test_queries import generate_test_queries

def load_test_queries(path: str = "src/E3/t2_retrieval/test_queries.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"⚠️ {path} no encontrado. Generando automáticamente...")
        df = generate_test_queries(output_path=path)
        return df
    return pd.read_csv(path)

def compute_mrr(relevant_ranks: List[int]) -> float:
    """Mean Reciprocal Rank: promedio de 1/rank del primer relevante."""
    if not relevant_ranks:
        return 0.0
    return np.mean([1.0 / rank for rank in relevant_ranks if rank > 0])

def compute_hit_at_k(relevant_ranks: List[int], k: int) -> float:
    """% de queries donde al menos un relevante está en los primeros k."""
    hits = sum(1 for r in relevant_ranks if r <= k)
    return hits / len(relevant_ranks) if relevant_ranks else 0.0

def evaluate_retriever(retriever, test_df: pd.DataFrame, k: int = 5) -> Dict:
    """Evalúa SBERT, TF-IDF y Hybrid."""
    methods = ["sbert", "tfidf", "hybrid"]
    results = {m: {"ranks": [], "hit1": [], "hit5": []} for m in methods}
    
    for _, row in test_df.iterrows():
        query = row["question"]
        relevant_chunks = eval(row["relevant_chunk_ids"])  # list of chunk_ids
        
        for method in methods:
            # Para hybrid con re-ranking por intención
            intent_weights = {}
            if "warning" in query.lower() or "safe" in query.lower():
                intent_weights = {"is_warning": 1.8}
            elif "step" in query.lower() or "how to" in query.lower():
                intent_weights = {"is_procedure": 1.5}
            
            df_results = retriever.search(
                query, 
                k=10, 
                method=method, 
                intent_weights=intent_weights
            )
            
            # Encontrar rank del primer relevante
            rank = None
            for _, res in df_results.iterrows():
                if res["chunk_id"] in relevant_chunks:
                    rank = res["rank"]
                    break
            rank = rank if rank is not None else 11  # >10 → no relevante en top-10
            
            results[method]["ranks"].append(rank)
    
    # Métricas finales
    metrics = {}
    for method in methods:
        ranks = results[method]["ranks"]
        metrics[method] = {
            "MRR@10": compute_mrr(ranks),
            "Hit@1": compute_hit_at_k(ranks, 1),
            "Hit@5": compute_hit_at_k(ranks, 5),
            "Avg_Rank_First_Relevant": np.mean([r for r in ranks if r <= 10] or [10])
        }
    
    return metrics