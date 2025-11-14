# experiments/t2_retrieval/run_retrieval.py
import pandas as pd
import numpy as np
from chunking import create_chunks_dataframe
from embedding_chunks import generate_chunk_embeddings
import retrieval
from evaluation import evaluate_retriever, load_test_queries

def main():
    # 1. Cargar documentos y crear chunks
    #df_docs = pd.read_pickle("data/processed/processed_documents.pkl")
    #chunks_df = create_chunks_dataframe(df_docs)
    chunks_df = pd.read_pickle("data/processed/chunks.pkl")
    
    # 2. Generar embeddings por chunk
    #sbert_emb = generate_chunk_embeddings(chunks_df)
    sbert_emb = np.load("data/embeddings/sbert_chunk_embeddings.npy")
    
    # 3. Inicializar retriever
    retriever = retrieval.HybridRetriever(chunks_df, sbert_emb)
    retriever.save_index()
    
    # 4. Evaluar
    test_df = load_test_queries()
    metrics = evaluate_retriever(retriever, test_df, k=5)
    
    # 5. Mostrar resultados
    print("\n📊 RESULTADOS DE RECUPERACIÓN SEMÁNTICA")
    print("=" * 50)
    for method, res in metrics.items():
        print(f"{method.upper():8} → MRR@10: {res['MRR@10']:.3f} | Hit@1: {res['Hit@1']:.3f} | Hit@5: {res['Hit@5']:.3f}")
    
    # Guardar
    import json
    with open("results/retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Métricas guardadas en results/retrieval_metrics.json")

if __name__ == "__main__":
    main()