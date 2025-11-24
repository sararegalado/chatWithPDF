import ollama
import time
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Clasificador de intenciones few-shot usando Ollama
class FewShotIntentClassifier:
    
    # Taxonomía de intenciones
    INTENTS = {
        "how_to": "The user asks for step-by-step instructions or procedures.",
        "spec_query": "The user asks for technical specifications, measurements, or values.",
        "diagnose": "The user reports a problem, symptom, or error code and seeks diagnosis.",
        "safety_check": "The user asks about risks, precautions, permissions, or safety conditions.",
        "summary": "The user asks for a summary of the manual",
        "other": "None of the above."
    }
    
    # Prompt template optimizado
    PROMPT_TEMPLATE = """You are a technical support assistant for manuals. Classify the user's question into exactly ONE of these intents:

    {intent_descriptions}

    Question: "{question}"
    Intent (one word only):

    Here are some examples: "{examples}"
    """

    # Inicializar el clasificador few-shot
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434",
                 num_predict: int = 12):
        
        self.model = model
        self.num_predict = num_predict
        self.host = host
        self.client = ollama.Client(host=host)
        
        print(f"Clasificador inicializado con modelo: {model}")
        print(f"Host: {host}")
    
    # Construir el prompt de clasificación
    def build_prompt(self, question: str) -> str:
        descriptions = "\n".join([f"- {k}: {v}" for k, v in self.INTENTS.items()])
        with open("src/E3/t1_intent_classification/ground_truth/few_shot_examples.json", "r", encoding="utf-8") as f:
            import json
            examples_data = json.load(f)
        examples = " | ".join([item["text"] for item in examples_data])
        return self.PROMPT_TEMPLATE.format(
            intent_descriptions=descriptions,
            question=question.strip().replace('"', "'"),
            examples=examples
        )
    
    # Clasificar la intención de una pregunta
    def classify_intent(self, question: str, verbose: bool = False) -> tuple[str, float]:
        prompt = self.build_prompt(question)
        start_time = time.time()
        
        try:
            # Llamada a Ollama
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": self.num_predict,
                    "seed": 42
                }
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            raw = response.response.strip().lower()
            
            if verbose:
                print(f"Respuesta raw: '{raw}'")
            
            # Extraer intención: primera palabra, limpiar puntuación
            words = raw.split()
            if not words:
                if verbose:
                    print(f"Respuesta vacía del modelo")
                return "other", inference_time
            
            intent = words[0].rstrip(".,;:!?'\"").strip()
            
            # Validar intención
            if intent in self.INTENTS:
                return intent, inference_time
            else:
                # Intentar recuperar con fuzzy match
                for valid in self.INTENTS:
                    if valid in raw:
                        if verbose:
                            print(f"Fuzzy match: '{intent}' → '{valid}'")
                        return valid, inference_time
                
                if verbose:
                    print(f"Intent no reconocido '{intent}', usando 'other'")
                return "other", inference_time
                
        except ollama.ResponseError as e:
            print(f"ResponseError: {e.error}")
            if "model not found" in str(e).lower():
                print(f"\nPara usar este modelo:\n    ollama pull {self.model}\n")
            return "other", 0.0
            
        except ollama.RequestError as e:
            print(f"RequestError: no se puede conectar a Ollama")
            print(f"   Verifica que Ollama esté ejecutándose en {self.host}")
            return "other", 0.0
            
        except AttributeError as e:
            print(f"AttributeError: respuesta inesperada del modelo - {e}")
            return "other", 0.0
            
        except Exception as e:
            print(f"Error inesperado: {type(e).__name__}: {e}")
            return "other", 0.0
    
    def verify_connection(self) -> bool:
        print("\nVerificando conexión con Ollama...")
        
        # Verificar conexión
        try:
            self.client.list()
            print(f"Ollama está accesible en {self.host}")
        except Exception as e:
            print(f"Error al conectar con Ollama: {e}")
            print("Asegúrate de que Ollama esté en ejecución")
            return False
        
        # Verificar modelo
        print(f"\nVerificando modelo '{self.model}'...")
        try:
            models = [m["model"] for m in self.client.list()["models"]]
            if any(self.model == m or self.model in m for m in models):
                print(f"Modelo '{self.model}' disponible")
                return True
            else:
                print(f"Modelo '{self.model}' NO encontrado")
                print(f"Ejecuta: ollama pull {self.model}")
                return False
        except Exception as e:
            print(f"Error al listar modelos: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        return {
            "model": self.model,
            "host": self.host,
            "num_predict": self.num_predict,
            "intents": list(self.INTENTS.keys()),
            "num_intents": len(self.INTENTS)
        }

    def evaluate_on_test_set(self, df_test: pd.DataFrame) -> dict:
        print(f"Evaluando clasificación few-shot con {self.model}...")
        
        predictions = []
        times = []
        
        for i, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Clasificando"):
            pred, inference_time = self.classify_intent(row["text"])
            predictions.append(pred)
            times.append(inference_time)
            
            # Mostrar ejemplos ocasionales
            if i % 10 == 0 and i > 0:
                print(f"\nEjemplo {i}: '{row['text'][:40]}...'")
                print(f"Real: {row['label']} → Pred: {pred} ({inference_time:.1f} ms)")
        
        # Métricas
        y_true = df_test["label"].tolist()
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
            "model": self.model,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "avg_inference_time_ms": avg_time,
            "predictions": y_pred,
            "true_labels": y_true,
            "present_labels": present_labels
        }
        
        # Mostrar resumen
        print(f"RESULTADOS ({self.model})")
        print(f"   Accuracy:       {acc:.3f}")
        print(f"   F1-macro:       {f1_macro:.3f}")
        print(f"   Tiempo promedio: {avg_time:.1f} ms/pregunta")
        
        return results
    

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], present_labels: List[str], save_path: str):

        labels = list(self.INTENTS.keys())
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
        plt.title("Few-Shot Intent Classification\nConfusion Matrix (qwen3:4b)", fontsize=14)
        plt.ylabel("True Intent")
        plt.xlabel("Predicted Intent")
        plt.tight_layout()
        
        os.makedirs("results", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matriz de confusión guardada en: {save_path}")
        plt.close()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear clasificador
    classifier = FewShotIntentClassifier(
        model="mistral:7b-instruct-q4_k_m",
        host="http://localhost:11434"
    )
    
    # Verificar conexión
    if not classifier.verify_connection():
        print("\n   No se puede continuar sin conexión a Ollama")
        exit(1)
    
    # Mostrar info del modelo
    print("\nInformación del clasificador:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # EVALUACIÓN
    # Cargar conjunto de evaluación
    import json
    with open("src/E3/t1_intent_classification/ground_truth/evaluation_dataset.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    df_eval = pd.DataFrame(eval_data)
    # Debug: ver columnas antes del rename
    print(f"\nColumnas originales: {df_eval.columns.tolist()}")
    print(f"Primeras filas:\n{df_eval.head()}")

    # Evaluar en el conjunto de prueba
    results = classifier.evaluate_on_test_set(df_eval)

    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        y_true=results["true_labels"],
        y_pred=results["predictions"],
        present_labels=results["present_labels"],
        save_path="results/cm_few_shot_mistral.png"
    )
