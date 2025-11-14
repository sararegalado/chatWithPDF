import pandas as pd
import re
from typing import List, Dict
import os

def smart_chunk_text(
    text: str,
    max_chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_size: int = 100
) -> List[str]:
    # Respetar estructura: títulos, listas, advertencias
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text).strip()

    # Split por señales estructurales típicas de manuales
    sections = re.split(
        r'(?=\n\s*(?:\d+\.\s+|[A-Z][A-Z\s]+:|CAUTION|WARNING|NOTE|Step\s+\d+:|Section\s+\d+))',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    chunks = []
    for section in sections:
        if len(section.split()) < min_chunk_size:
            if section.strip():
                chunks.append(section.strip())
            continue
        
        # Si es largo, dividir con solapamiento
        words = section.split()
        start = 0
        while start < len(words):
            end = start + max_chunk_size
            chunk = ' '.join(words[start:end])
            
            # Evitar cortar frases críticas
            last_break = max(chunk.rfind('.'), chunk.rfind(':'), chunk.rfind('\n'))
            if last_break > max_chunk_size * 0.7 and last_break > 20:
                chunk = chunk[:last_break + 1]
            
            if len(chunk.split()) >= min_chunk_size:
                chunks.append(chunk.strip())
            start += max_chunk_size - overlap
    
    return chunks


# Generar dataframe de chunks con metadatos
def create_chunks_dataframe(df_docs: pd.DataFrame) -> pd.DataFrame:
    records = []
    chunk_id = 0
    
    for _, doc in df_docs.iterrows():
        text = doc['clean_text']
        if not isinstance(text, str) or len(text) < 20:
            continue
            
        chunks = smart_chunk_text(text, max_chunk_size=400, overlap=50)
        
        for i, chunk in enumerate(chunks):
            # Features estructurales (para re-ranking futuro)
            first_line = chunk.split('\n')[0].strip()
            record = {
                'chunk_id': chunk_id,
                'doc_id': doc.name,  # índice original
                'filename': doc['filename'],
                'chunk_text': chunk,
                'chunk_index': i,
                'is_warning': bool(re.search(r'⚠|WARNING|CAUTION|DANGER', chunk, re.I)),
                'is_procedure': bool(re.match(r'^\d+\.\s*|\bStep\s+\d+', first_line, re.I)),
                'has_table': chunk.count('|') >= 4 or ':\s*\d' in chunk,
                'num_words': len(chunk.split())
            }
            records.append(record)
            chunk_id += 1
    
    chunks_df = pd.DataFrame(records)
    print(f"✅ Generados {len(chunks_df)} chunks de {len(df_docs)} documentos.")
    return chunks_df


if __name__ == "__main__":
    df_docs = pd.read_pickle("data/processed/processed_documents.pkl")
    chunks_df = create_chunks_dataframe(df_docs)
    os.makedirs("data/processed/", exist_ok=True)
    chunks_df.to_pickle("data/processed/chunks.pkl")
    print("💾 Chunks guardados en data/processed/chunks.pkl")