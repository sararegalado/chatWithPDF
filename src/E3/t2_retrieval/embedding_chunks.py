# experiments/t2_retrieval/embedding_chunks.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# Funcion para generar embeddings del dataframe de chunks con BERT
def generate_chunk_embeddings(
    chunks_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = "cpu"
) -> np.ndarray:
    print(f"🔹 Generando embeddings de chunks con {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    
    texts = chunks_df['chunk_text'].tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # crucial para cosine similarity con FAISS
    )
    
    print(f"✅ Embeddings generados: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    chunks_df = pd.read_pickle("data/processed/chunks.pkl")
    sbert_chunk_emb = generate_chunk_embeddings(chunks_df)
    
    os.makedirs("data/embeddings/", exist_ok=True)
    np.save("data/embeddings/sbert_chunk_embeddings.npy", sbert_chunk_emb)
    print("💾 Embeddings por chunk guardados.")