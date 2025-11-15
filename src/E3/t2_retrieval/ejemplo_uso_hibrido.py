import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Añadir rutas
sys.path.append(str(Path(__file__).parent))
from retrieval import HybridRetriever

# Ejemplo del sistema hibrido de retrieval
def ejemplo_completo():

    # 1. Cargar base de conocimiento
    print("\nPaso 1: Cargando base de conocimiento...")
    try:
        chunks_df = pd.read_pickle("data/processed/chunks.pkl")
        embeddings = np.load("data/embeddings/sbert_chunk_embeddings.npy")
        print(f"{len(chunks_df)} chunks cargados")
        print(f"{chunks_df['filename'].nunique()} PDFs únicos")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 2. Inicializar retriever
    print("\nPaso 2: Inicializando retriever...")
    retriever = HybridRetriever(chunks_df, embeddings)
    
    # 3. ESCENARIO 1: Usuario NO carga PDF
    print("\nESCENARIO 1: Usuario pregunta SIN cargar PDF específico\n")    
    pregunta_1 = "How do I clean the filter?"
    print(f"\nPregunta del usuario: '{pregunta_1}'")
    print("Sistema busca en TODA la base de conocimiento...")
    resultados_1 = retriever.search(
        query=pregunta_1,
        filename=None,  # None = búsqueda global
        k=5,
        method="hybrid"
    )
    
    print(f"\nEncontrados {len(resultados_1)} chunks relevantes:\n")
    for _, row in resultados_1.iterrows():
        print(f"\nRank {row['rank']} - [{row['filename']}] (Score: {row['score']:.3f})")
        print(f"   📝 {row['text_preview']}")
    
    
    # 4. ESCENARIO 2: Usuario CARGA PDF
    print("\nESCENARIO 2: Usuario carga un PDF específico\n")
    
    # Simular: usuario selecciona/carga un PDF
    pdf_seleccionado = retriever.get_available_pdfs()[5]
    pregunta_2 = "What are the safety warnings?"
    
    print(f"\nUsuario carga: '{pdf_seleccionado}'")
    print(f"Pregunta del usuario: '{pregunta_2}'")
    print(f"Sistema busca SOLO en ese documento...")
    
    resultados_2 = retriever.search(
        query=pregunta_2,
        filename=pdf_seleccionado,  # Filtrar por documento
        k=5,
        method="hybrid",
        intent_weights={"is_warning": 1.5}  # Priorizar advertencias
    )
    
    if not resultados_2.empty:
        print(f"\nEncontrados {len(resultados_2)} chunks en '{pdf_seleccionado}':")
        print("-" * 80)
        for _, row in resultados_2.iterrows():
            print(f"\nRank {row['rank']} (Score: {row['score']:.3f})")
            print(f"   {row['text_preview']}")
            if row['is_warning']:
                print("   Contiene advertencias de seguridad")
    else:
        print(f"\nNo se encontraron resultados en '{pdf_seleccionado}'")
    
    # 5. COMPARACIÓN DE MÉTODOS
    print("\nComparación de métodos (SBERT vs TF-IDF vs Hybrid) basado en base de conocimiento global\n")

    pregunta_3 = "installation steps"
    print(f"\nPregunta: '{pregunta_3}'")
    print("Probando los 3 métodos...\n")
    
    for metodo in ["sbert", "tfidf", "hybrid"]:
        resultados = retriever.search(
            query=pregunta_3,
            filename=None,
            k=3,
            method=metodo
        )
        
        print(f"\n{metodo.upper()}:")
        print("-" * 40)
        if not resultados.empty:
            for _, row in resultados.iterrows():
                print(f"  {row['rank']}. [{row['filename']}] (score: {row['score']:.3f})")
        else:
            print("  Sin resultados")
    
    # 6. Estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS DEL SISTEMA")
    print("="*80)
    
    stats_global = retriever.get_stats(filename=None)
    print("\nBase de conocimiento completa:")
    for key, value in stats_global.items():
        print(f"   • {key}: {value}")
    
    stats_doc = retriever.get_stats(filename=pdf_seleccionado)
    print(f"\nDocumento '{pdf_seleccionado}':")
    for key, value in stats_doc.items():
        print(f"   • {key}: {value}")
    
    
    


if __name__ == "__main__":
    ejemplo_completo()
