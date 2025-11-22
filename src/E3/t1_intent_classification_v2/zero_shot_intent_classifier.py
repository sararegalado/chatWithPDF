import ollama
import time
from typing import Dict, Optional


class ZeroShotIntentClassifier:
    """
    Clasificador de intenciones usando Zero-Shot Learning con Ollama LLMs.
    """
    
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
    Intent (one word only):"""
    
    # Inicializar el clasificador zero-shot
    def __init__(self, model: str = "llama3", host: str = "http://192.168.0.124:11434", 
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
        return self.PROMPT_TEMPLATE.format(
            intent_descriptions=descriptions,
            question=question.strip().replace('"', "'")
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
        """
        Obtiene información del modelo y configuración.
        
        Returns:
            Diccionario con información del clasificador
        """
        return {
            "model": self.model,
            "host": self.host,
            "num_predict": self.num_predict,
            "intents": list(self.INTENTS.keys()),
            "num_intents": len(self.INTENTS)
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear clasificador
    classifier = ZeroShotIntentClassifier(
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
    # Cargar dataset
    print("\nEvaluando classifier con el dataset sintetico...")
    with open("src/E3/t1_intent_classification_v2/ground_truth/intents_dataset.json", "r", encoding="utf-8") as f:
        import json
        dataset = json.load(f)
    correct, total = 0, 0
    for item in dataset:
        question = item["text"]
        true_intent = item["label"]
        predicted_intent, time_ms = classifier.classify_intent(question, verbose=False)
        is_correct = (predicted_intent == true_intent)
        correct += int(is_correct)
        total += 1
        print(f"\nPregunta: '{question}'")
        print(f"   → Predicho: {predicted_intent}, Verdadero: {true_intent} {'✅' if is_correct else '❌'} ({time_ms:.1f} ms)")
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\nPrecisión total: {accuracy:.2f}% ({correct}/{total})")
    
    # Ejemplos de clasificación
    print("\n" + "="*60)
    print("EJEMPLOS DE CLASIFICACIÓN")
    print("="*60)
    
    test_questions = [
        "How do I reset the device?",
        "What is the maximum voltage?",
        "The screen is flashing red",
        "Is it safe to use in the rain?",
        "Summarize the manual for me",
        "What is the weather today?"
    ]
    
    for question in test_questions:
        intent, time_ms = classifier.classify_intent(question, verbose=False)
        print(f"\n'{question}'")
        print(f"   → Intent: {intent} ({time_ms:.1f} ms)")