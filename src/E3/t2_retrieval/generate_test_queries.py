# src/E3/t2_retrieval/generate_test_queries.py
import pandas as pd
import numpy as np
import os
import re
from typing import List

def generate_test_queries(chunks_df_path: str = "data/processed/chunks.pkl",
                          output_path: str = "src/E3/t2_retrieval/test_queries.csv"):
    # Cargar chunks
    chunks_df = pd.read_pickle(chunks_df_path)
    print(f"✅ Cargados {len(chunks_df)} chunks")

    # Separar chunks por tipo funcional (usando tus columnas existentes)
    warning_chunks = chunks_df[chunks_df['is_warning']]
    procedure_chunks = chunks_df[chunks_df['is_procedure']]
    spec_chunks = chunks_df[chunks_df['has_table']]
    generic_chunks = chunks_df[~(chunks_df['is_warning'] | chunks_df['is_procedure'] | chunks_df['has_table'])]

    # Plantillas de preguntas realistas
    templates = {
        'how_to': [
            "How do I {action}?",
            "What are the steps to {action}?",
            "How to {action} the device?",
            "Instructions for {action}.",
            "Procedure to {action}."
        ],
        'spec_query': [
            "What is the {spec}?",
            "What are the {spec} of the device?",
            "Maximum {spec} allowed?",
            "Value of {spec}?",
            "Does it support {spec}?"
        ],
        'safety_check': [
            "Is it safe to {action}?",
            "Do I need protection to {action}?",
            "Are there risks when {action}?",
            "Can I {action} while powered?",
            "Is grounding required for {action}?"
        ],
        'diagnose': [
            "What does error {code} mean?",
            "Why does the device {symptom}?",
            "Device shows {symptom}. What should I do?",
            "Error {code} appears on startup.",
            "The unit {symptom}. Help?"
        ]
    }

    # Acciones y specs realistas (ajustables)
    actions = ["reset the thermostat", "replace the battery", "clean the filter",
               "install the firmware", "enter diagnostic mode", "mount the bracket"]
    specs = ["maximum voltage", "weight", "dimensions", "operating temperature",
             "power consumption", "IP rating"]
    symptoms = ["overheats", "won't turn on", "flashes red", "leaks water",
                "makes noise", "disconnects randomly"]
    codes = ["E12", "ERR-404", "CODE-77", "F9", "ALARM-3"]

    test_queries = []

    # 1. how_to → procedure_chunks
    for i, action in enumerate(actions):
        q = np.random.choice(templates['how_to']).format(action=action)
        # Buscar chunks que contengan la acción (búsqueda aproximada)
        candidates = procedure_chunks[
            procedure_chunks['chunk_text'].str.contains(action.split()[0], case=False, na=False)
        ]
        if len(candidates) == 0:
            candidates = procedure_chunks
        relevant = candidates.head(3)['chunk_id'].tolist() if not candidates.empty else [0]
        test_queries.append({"question": q, "relevant_chunk_ids": str(relevant), "intent": "how_to"})

    # 2. spec_query → spec_chunks
    for spec in specs:
        q = np.random.choice(templates['spec_query']).format(spec=spec)
        candidates = spec_chunks[
            spec_chunks['chunk_text'].str.contains(spec.split()[0], case=False, na=False)
        ]
        if len(candidates) == 0:
            candidates = spec_chunks
        relevant = candidates.head(2)['chunk_id'].tolist() if not candidates.empty else [0]
        test_queries.append({"question": q, "relevant_chunk_ids": str(relevant), "intent": "spec_query"})

    # 3. safety_check → warning_chunks
    for action in ["open the cover", "touch the terminals", "operate in humid environment"]:
        q = np.random.choice(templates['safety_check']).format(action=action)
        candidates = warning_chunks[
            warning_chunks['chunk_text'].str.contains("safe|risk|danger|caution|warning", case=False)
        ]
        relevant = candidates.head(3)['chunk_id'].tolist() if not candidates.empty else [0]
        test_queries.append({"question": q, "relevant_chunk_ids": str(relevant), "intent": "safety_check"})

    # 4. diagnose → buscar por síntomas/códigos
    for code in codes:
        q = f"What does error {code} mean?"
        candidates = chunks_df[
            chunks_df['chunk_text'].str.contains(code, case=False, na=False)
        ]
        relevant = candidates.head(2)['chunk_id'].tolist() if not candidates.empty else [
            warning_chunks.iloc[0]['chunk_id'] if not warning_chunks.empty else 0
        ]
        test_queries.append({"question": q, "relevant_chunk_ids": str(relevant), "intent": "diagnose"})

    for symptom in symptoms[:3]:
        q = f"Why does the device {symptom}?"
        candidates = chunks_df[
            chunks_df['chunk_text'].str.contains(symptom.split()[0], case=False, na=False)
        ]
        relevant = candidates.head(2)['chunk_id'].tolist() if not candidates.empty else [0]
        test_queries.append({"question": q, "relevant_chunk_ids": str(relevant), "intent": "diagnose"})

    # Crear DataFrame
    df = pd.DataFrame(test_queries)
    print(f"✅ Generadas {len(df)} preguntas de prueba")

    # Asegurar que el directorio existe
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Guardar con comillas y escaping correctos
    df.to_csv(output_path, index=False, quoting=1)  # quoting=1 → comillas en todos los campos
    print(f"💾 Guardado en {output_path}")

    return df

if __name__ == "__main__":
    generate_test_queries()