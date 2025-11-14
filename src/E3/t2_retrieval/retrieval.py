# experiments/t2_retrieval/retrieval.py
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, chunks_df: pd.DataFrame, sbert_embeddings: np.ndarray):
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.sbert_embeddings = sbert_embeddings.astype('float32')
        self.texts = self.chunks_df['chunk_text'].tolist()
        
        # 1. Índice FAISS para SBERT (cosine similarity)
        self.dim = self.sbert_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dim)  # inner product = cosine si normalizados
        faiss.normalize_L2(self.sbert_embeddings)
        self.faiss_index.add(self.sbert_embeddings)
        
        # 2. TF-IDF (shallow baseline)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)
        
        print(f"✅ Retriever inicializado: {len(self.texts)} chunks")

    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "sbert",  # "sbert", "tfidf", "hybrid"
        alpha: float = 0.7,     # peso SBERT en hybrid
        intent_weights: dict = None  # ej: {"is_warning": 1.5}
    ) -> pd.DataFrame:
        if method == "sbert":
            # Embedding de la query
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
            
            # Búsqueda FAISS
            distances, indices = self.faiss_index.search(query_emb, k)
            scores = distances[0]
            chunk_ids = indices[0]
        
        elif method == "tfidf":
            query_vec = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_k = np.argsort(similarities)[::-1][:k]
            chunk_ids = top_k
            scores = similarities[top_k]
        
        elif method == "hybrid":
            # SBERT scores
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
            d_sbert, i_sbert = self.faiss_index.search(q_emb, k * 3)  # traer más para re-rank
            
            # TF-IDF scores
            q_tfidf = self.tfidf_vectorizer.transform([query])
            sim_tfidf = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
            
            # Combinar (solo sobre los top de SBERT)
            combined_scores = []
            for idx in i_sbert[0]:
                score = alpha * d_sbert[0][list(i_sbert[0]).index(idx)] + (1 - alpha) * sim_tfidf[idx]
                # Aplicar intent_weights si existen
                if intent_weights:
                    for col, weight in intent_weights.items():
                        if self.chunks_df.iloc[idx][col]:
                            score *= weight
                combined_scores.append((idx, score))
            
            # Ordenar y tomar top-k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            chunk_ids = [idx for idx, _ in combined_scores[:k]]
            scores = [score for _, score in combined_scores[:k]]
        
        # Construir resultado
        results = []
        for i, (cid, score) in enumerate(zip(chunk_ids, scores)):
            if cid >= len(self.chunks_df):
                continue
            chunk = self.chunks_df.iloc[cid]
            results.append({
                'rank': i + 1,
                'chunk_id': int(cid),
                'filename': chunk['filename'],
                'text': chunk['chunk_text'][:150] + "...",
                'score': float(score),
                'is_warning': chunk['is_warning'],
                'is_procedure': chunk['is_procedure']
            })
        
        return pd.DataFrame(results)

    def save_index(self, path: str = "data/retrieval/faiss.index"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.faiss_index, path)
        print(f"💾 Índice FAISS guardado en {path}")