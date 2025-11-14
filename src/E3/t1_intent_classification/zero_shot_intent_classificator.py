#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Intent Classification for Technical QA
Model: qwen3:4b via Ollama (local, private, free)
"""

import ollama
import pandas as pd
import numpy as np
import os
import json
import time
from typing import List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ----------------------------------------
# 1. CONFIGURACIÓN
# ----------------------------------------
MODEL = "llama3"  # ligero, rápido, preciso para texto. Usa "qwen3-vl:4b" si necesitas multimodal.
NUM_PREDICT = 12
client = ollama.Client(host="http://192.168.0.124:11434")  # ← PON AQUÍ LA IP DE TU OTRO PC


# Taxonomía de intenciones (ajustable)
INTENTS = {
    "how_to": "The user asks for step-by-step instructions or procedures.",
    "spec_query": "The user asks for technical specifications, measurements, or values.",
    "diagnose": "The user reports a problem, symptom, or error code and seeks diagnosis.",
    "safety_check": "The user asks about risks, precautions, permissions, or safety conditions.",
    "summary": "The user asks for a summary of the manual",
    "other": "None of the above."
}

# Prompt optimizado para zero-shot con qwen3:4b
PROMPT_TEMPLATE = """You are a technical support assistant for manuals. Classify the user's question into exactly ONE of these intents:

{intent_descriptions}

Question: "{question}"
Intent (one word only):"""


def build_prompt(question: str) -> str:
    descriptions = "\n".join([f"- {k}: {v}" for k, v in INTENTS.items()])
    return PROMPT_TEMPLATE.format(
        intent_descriptions=descriptions,
        question=question.strip().replace('"', "'")
    )


# ----------------------------------------
# 2. FUNCIÓN DE CLASIFICACIÓN
# ----------------------------------------
def classify_intent_zero_shot(question: str, model: str = MODEL) -> str:
    prompt = build_prompt(question)
    
    try:
        # Llamada a Ollama
        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "num_predict": NUM_PREDICT,
                "seed": 42
            }
        )
        raw = response.response.strip().lower()
        
        # Extraer intención: primera palabra, limpiar puntuación
        words = raw.split()
        if not words:
            print(f"⚠️ Respuesta vacía del modelo para: '{question[:40]}...'")
            return "other"
        
        intent = words[0].rstrip(".,;:!?'\"").strip()
        
        # Validar
        if intent in INTENTS:
            return intent
        else:
            # Intentar recuperar con fuzzy match
            for valid in INTENTS:
                if valid in raw:
                    return valid
            
            print(f"⚠️ Intent no reconocido '{intent}' en respuesta: '{raw[:50]}...'")
            return "other"
            
    except client.ResponseError as e:
        print(f"⚠️ ResponseError: {e.error}")
        if "model not found" in str(e).lower():
            print(f"\n💡 Para usar este script:\n    ollama pull {MODEL}\n")
        return "other"
    except client.RequestError as e:
        print(f"⚠️ RequestError: no se puede conectar a Ollama (¿está en ejecución?)")
        return "other"
    except AttributeError as e:
        print(f"⚠️ AttributeError: respuesta inesperada del modelo - {e}")
        print(f"   Tipo de respuesta: {type(response)}")
        return "other"
    except Exception as e:
        print(f"⚠️ Error inesperado: {type(e).__name__}: {e}")
        return "other"


# ----------------------------------------
# 3. CONJUNTO DE PRUEBA (30 preguntas reales simuladas)
# ----------------------------------------
TEST_SET = [
    # how_to (8)
    ("How do I reset the thermostat?", "how_to"),
    ("What are the steps to replace the battery?", "how_to"),
    ("How to calibrate the sensor?", "how_to"),
    ("Instructions for installing the firmware update.", "how_to"),
    ("How do I pair the device with my phone?", "how_to"),
    ("What is the procedure to clean the filter?", "how_to"),
    ("How to enter diagnostic mode?", "how_to"),
    ("Steps to mount the bracket on the wall.", "how_to"),
    
    # spec_query (7)
    ("What is the maximum voltage?", "spec_query"),
    ("Weight of model X200?", "spec_query"),
    ("What are the dimensions of the unit?", "spec_query"),
    ("Input frequency range?", "spec_query"),
    ("Is the operating temperature -10°C to 50°C?", "spec_query"),
    ("What is the IP rating?", "spec_query"),
    ("Power consumption in standby mode?", "spec_query"),
    
    # diagnose (8)
    ("Why won't the device turn on?", "diagnose"),
    ("What does error code E12 mean?", "diagnose"),
    ("The screen is flashing red. What should I do?", "diagnose"),
    ("Device overheats after 5 minutes. Help?", "diagnose"),
    ("Error ERR-404 appears on startup.", "diagnose"),
    ("No response when pressing buttons.", "diagnose"),
    ("Water leak from bottom panel.", "diagnose"),
    ("Intermittent Wi-Fi connection.", "diagnose"),
    
    # safety_check (7)
    ("Is it safe to open the cover while powered?", "safety_check"),
    ("Do I need to wear gloves for this procedure?", "safety_check"),
    ("Can this device be used in a bathroom?", "safety_check"),
    ("Is grounding required?", "safety_check"),
    ("Are there any radiation hazards?", "safety_check"),
    ("Can children operate this machine?", "safety_check"),
    ("Is it fire-resistant?", "safety_check")
]

def load_test_set() -> pd.DataFrame:
    return pd.DataFrame(TEST_SET, columns=["question", "true_intent"])


# ----------------------------------------
# 4. EVALUACIÓN
# ----------------------------------------
def evaluate_on_test_set(df_test: pd.DataFrame, model: str = MODEL) -> dict:
    print(f"🔍 Evaluando clasificación zero-shot con {model}...")
    print("=" * 60)
    
    predictions = []
    times = []
    
    for i, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Clasificando"):
        start = time.time()
        pred = classify_intent_zero_shot(row["question"], model=model)
        elapsed = (time.time() - start) * 1000  # ms
        predictions.append(pred)
        times.append(elapsed)
        
        # Mostrar ejemplos ocasionales
        if i % 10 == 0 and i > 0:
            print(f"\n📊 Ejemplo {i}: '{row['question'][:40]}...'")
            print(f"   Real: {row['true_intent']} → Pred: {pred} ({elapsed:.1f} ms)")
    
    # Métricas
    y_true = df_test["true_intent"].tolist()
    y_pred = predictions
    
    # Calcular solo sobre clases que aparecen en el test set
    present_labels = sorted(set(y_true))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=present_labels,
        zero_division=0
    )
    
    avg_time = np.mean(times)
    
    results = {
        "model": model,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "avg_inference_time_ms": avg_time,
        "predictions": y_pred,
        "true_labels": y_true,
        "present_labels": present_labels # Guardar para la matriz de confusión
    }
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print(f"✅ RESULTADOS ({model})")
    print(f"   Accuracy:       {acc:.3f}")
    print(f"   F1-macro:       {f1_macro:.3f}")
    print(f"   Tiempo promedio: {avg_time:.1f} ms/pregunta")
    print("=" * 60)
    
    return results


# ----------------------------------------
# 5. VISUALIZACIÓN
# ----------------------------------------
def plot_confusion_matrix(y_true: List[str], y_pred: List[str], present_labels: List[str], save_path: str = "results/cm_zero_shot.png"):
    labels = list(INTENTS.keys())
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        cbar_kws={'shrink': 0.8}
    )
    plt.title("Zero-Shot Intent Classification\nConfusion Matrix (qwen3:4b)", fontsize=14)
    plt.ylabel("True Intent")
    plt.xlabel("Predicted Intent")
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Matriz de confusión guardada en {save_path}")
    plt.close()


# ----------------------------------------
# 6. EJECUCIÓN PRINCIPAL
# ----------------------------------------
def main():
    # Verificar Ollama
    print("🧪 Verificando conexión con Ollama...")
    try:
        client.list()
        print("✅ Ollama está accesible.")
    except Exception as e:
        print(f"❌ Error al conectar con Ollama: {e}")
        print("💡 Asegúrate de que Ollama esté en ejecución (app abierta o 'ollama serve' en terminal).")
        return
    
    # Verificar modelo
    print(f"🔍 Verificando modelo '{MODEL}'...")
    try:
        models = [m["model"] for m in client.list()["models"]]
        if any(MODEL == m or MODEL in m for m in models):
            print(f"✅ Modelo '{MODEL}' disponible.")
        else:
            print(f"⚠️ Modelo '{MODEL}' NO encontrado.")
            print(f"💡 Ejecuta: ollama pull {MODEL}")
            return
    except Exception as e:
        print(f"❌ Error al listar modelos: {e}")
        return
    
    # Cargar y evaluar
    df_test = load_test_set()
    results = evaluate_on_test_set(df_test, model=MODEL)
    
    # Guardar resultados detallados
    df_results = df_test.copy()
    df_results["predicted_intent"] = results["predictions"]
    df_results["is_correct"] = df_results["true_intent"] == df_results["predicted_intent"]
    
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/zero_shot_intent_results.csv", index=False)
    print("💾 Resultados detallados guardados en results/zero_shot_intent_results.csv")
    
    # Métricas JSON
    with open("results/zero_shot_metrics.json", "w") as f:
        json.dump({
            "model": MODEL,
            "accuracy": float(results["accuracy"]),
            "f1_macro": float(results["f1_macro"]),
            "avg_inference_time_ms": float(results["avg_inference_time_ms"]),
            "test_set_size": len(df_test)
        }, f, indent=2)
    
    # Visualización
    plot_confusion_matrix(results["true_labels"], results["predictions"], results["present_labels"])
    
    # Mostrar errores típicos
    errors = df_results[~df_results["is_correct"]]
    if not errors.empty:
        print("\n🔍 Errores comunes:")
        for _, row in errors.head(5).iterrows():
            print(f"  ❌ '{row['question'][:50]}...'")
            print(f"      Real: {row['true_intent']} → Pred: {row['predicted_intent']}")


if __name__ == "__main__":
    main()