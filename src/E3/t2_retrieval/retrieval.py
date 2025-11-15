# src/E3/t2_retrieval/retrieval.py
import numpy as np
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List

class HybridRetriever:
    """
    Sistema de Retrieval Híbrido con dos modos:
    
    1. BASE DE CONOCIMIENTO GLOBAL (filename=None):
       - Busca en TODOS los chunks de TODOS los PDFs
       - Útil cuando el usuario no tiene un documento específico
    
    2. DOCUMENTO ESPECÍFICO (filename="mi_manual.pdf"):
       - Filtra solo los chunks de ese PDF
       - Útil cuando el usuario carga/selecciona un documento concreto
    """
    
    def __init__(
        self,
        chunks_df: pd.DataFrame,
        chunk_embeddings: np.ndarray
    ):
        """
        Args:
            chunks_df: DataFrame con columnas ['filename', 'chunk_text', ...]
            chunk_embeddings: Array de embeddings (SBERT) para cada chunk
        """
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.sbert_embeddings = chunk_embeddings.astype('float32')
        self.dim = self.sbert_embeddings.shape[1]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # FAISS index global (todos los chunks)
        self.global_index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(self.sbert_embeddings)
        self.global_index.add(self.sbert_embeddings)
        
        # TF-IDF global
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.chunks_df['chunk_text'].tolist()
        )
        
        # Metadata
        self.available_pdfs = sorted(self.chunks_df['filename'].unique().tolist())
        
        print(f"✅ HybridRetriever inicializado:")
        print(f"   📚 {len(self.chunks_df)} chunks totales")
        print(f"   📄 {len(self.available_pdfs)} PDFs únicos")
        print(f"   🔍 Embeddings shape: {self.sbert_embeddings.shape}")

    def search(
        self,
        query: str,
        filename: Optional[str] = None,
        k: int = 5,
        method: str = "hybrid",
        alpha: float = 0.7,
        intent_weights: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Búsqueda con filtrado opcional por documento.
        
        Args:
            query: Pregunta del usuario
            filename: Nombre del PDF (None = buscar en TODA la base)
            k: Top-K resultados a devolver
            method: "sbert", "tfidf" o "hybrid"
            alpha: Peso SBERT en hybrid (0.7 = 70% SBERT, 30% TF-IDF)
            intent_weights: Dict para re-ranking (ej: {"is_warning": 1.5})
        
        Returns:
            DataFrame con columnas: rank, chunk_id, filename, text_preview, 
                                   full_text, score, search_scope
        """
        
        # Determinar scope y filtrar chunks
        if filename:
            # Modo: documento específico
            mask = self.chunks_df['filename'] == filename
            allowed_indices = self.chunks_df[mask].index.tolist()
            
            if not allowed_indices:
                print(f"⚠️ No se encontraron chunks para '{filename}'")
                return pd.DataFrame()
            
            search_scope = f"documento '{filename}'"
            print(f"🔍 Buscando en {search_scope} ({len(allowed_indices)} chunks)")
        else:
            # Modo: base de conocimiento global
            allowed_indices = list(range(len(self.chunks_df)))
            search_scope = "toda la base de conocimiento"
            print(f"🌐 Buscando en {search_scope} ({len(allowed_indices)} chunks)")
        
        # Ejecutar búsqueda según método
        if method == "sbert":
            scores, chunk_ids = self._search_sbert_filtered(query, allowed_indices, k)
        elif method == "tfidf":
            scores, chunk_ids = self._search_tfidf_filtered(query, allowed_indices, k)
        elif method == "hybrid":
            scores, chunk_ids = self._search_hybrid_filtered(
                query, allowed_indices, k, alpha, intent_weights
            )
        else:
            raise ValueError(f"Método '{method}' no reconocido. Usa: 'sbert', 'tfidf', 'hybrid'")
        
        # Construir DataFrame de resultados
        results = []
        for rank, (cid, score) in enumerate(zip(chunk_ids, scores), 1):
            if cid >= len(self.chunks_df):
                continue
            
            chunk = self.chunks_df.iloc[cid]
            results.append({
                'rank': rank,
                'chunk_id': int(cid),
                'filename': chunk['filename'],
                'text_preview': chunk['chunk_text'][:150] + "...",
                'full_text': chunk['chunk_text'],
                'score': float(score),
                'search_scope': search_scope,
                'is_warning': chunk.get('is_warning', False),
                'is_procedure': chunk.get('is_procedure', False)
            })
        
        return pd.DataFrame(results)

    def _search_sbert_filtered(
        self, 
        query: str, 
        allowed_indices: List[int], 
        k: int
    ) -> tuple:
        """Búsqueda SBERT sobre subset de chunks."""
        q_emb = self.model.encode([query], normalize_embeddings=True).astype('float32')
        
        # Crear índice temporal con solo los chunks permitidos
        filtered_embs = self.sbert_embeddings[allowed_indices].astype('float32')
        temp_index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(filtered_embs)
        temp_index.add(filtered_embs)
        
        # Buscar
        distances, indices = temp_index.search(q_emb, min(k, len(allowed_indices)))
        
        # Mapear índices locales → globales
        global_indices = [allowed_indices[i] for i in indices[0]]
        return distances[0], global_indices

    def _search_tfidf_filtered(
        self, 
        query: str, 
        allowed_indices: List[int], 
        k: int
    ) -> tuple:
        """Búsqueda TF-IDF sobre subset de chunks."""
        q_vec = self.tfidf_vectorizer.transform([query])
        
        # Calcular similitud solo con chunks permitidos
        filtered_matrix = self.tfidf_matrix[allowed_indices]
        similarities = cosine_similarity(q_vec, filtered_matrix).flatten()
        
        # Top-K
        top_k_local = np.argsort(similarities)[::-1][:k]
        global_indices = [allowed_indices[i] for i in top_k_local]
        scores = similarities[top_k_local]
        
        return scores, global_indices

    def _search_hybrid_filtered(
        self,
        query: str,
        allowed_indices: List[int],
        k: int,
        alpha: float,
        intent_weights: Optional[dict] = None
    ) -> tuple:
        """Búsqueda híbrida (SBERT + TF-IDF) sobre subset de chunks."""
        # SBERT scores (expandir a k*3 para tener más candidatos)
        sbert_scores, sbert_indices = self._search_sbert_filtered(
            query, allowed_indices, min(k * 3, len(allowed_indices))
        )
        
        # TF-IDF scores (calcular para todos los chunks)
        q_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
        
        # Combinar scores
        combined = []
        for idx, sbert_score in zip(sbert_indices, sbert_scores):
            # Score híbrido
            score = alpha * sbert_score + (1 - alpha) * tfidf_similarities[idx]
            
            # Re-ranking por intención (si existe)
            if intent_weights:
                for col, weight in intent_weights.items():
                    if col in self.chunks_df.columns and self.chunks_df.iloc[idx].get(col, False):
                        score *= weight
            
            combined.append((idx, score))
        
        # Ordenar y tomar top-K
        combined.sort(key=lambda x: x[1], reverse=True)
        final_indices = [idx for idx, _ in combined[:k]]
        final_scores = [score for _, score in combined[:k]]
        
        return final_scores, final_indices

    def get_available_pdfs(self) -> List[str]:
        """Retorna lista de PDFs disponibles en la base de conocimiento."""
        return self.available_pdfs

    def get_stats(self, filename: Optional[str] = None) -> dict:
        """Estadísticas de chunks (global o por documento)."""
        if filename:
            mask = self.chunks_df['filename'] == filename
            df = self.chunks_df[mask]
            scope = f"documento '{filename}'"
        else:
            df = self.chunks_df
            scope = "base de conocimiento completa"
        
        return {
            'scope': scope,
            'total_chunks': len(df),
            'unique_pdfs': df['filename'].nunique(),
            'avg_chunk_length': df['chunk_text'].str.len().mean(),
            'warnings_detected': df.get('is_warning', pd.Series([False])).sum(),
            'procedures_detected': df.get('is_procedure', pd.Series([False])).sum()
        }