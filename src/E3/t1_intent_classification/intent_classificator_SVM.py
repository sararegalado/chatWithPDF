import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import pickle

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Intent classifier usando Support Vector Machines (SVM) con embeddings de spaCy
class SVMIntentClassifier:

    # Test set
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
    
    def __init__(self, spacy_model: str = "en_core_web_md", C: float = 1.0):
        
        # Modelo de spaCy
        print(f"🔧 Cargando modelo de spaCy: {spacy_model}...")
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"Modelo cargado. Dimensión de vectores: {self.nlp.vocab.vectors_length}")
        except OSError:
            print(f"Modelo '{spacy_model}' no encontrado.")
            print(f"Instala con: python -m spacy download {spacy_model}")
            raise
        
        self.clf = SVC(C=C, kernel='rbf', probability=True)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.vector_dim = self.nlp.vocab.vectors_length
        
    # Convierte oraciones a embeddings usando spaCy
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        n_sentences = len(sentences)
        X = np.zeros((n_sentences, self.vector_dim))
        
        for idx, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            X[idx, :] = doc.vector
            
        return X
    
    # Entrenar el clasificador SVM
    def train(self, X_train: List[str], y_train: List[str], 
              validation_split: float = 0.2) -> dict:
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO SVM")
        print("="*60)
        
        # Split train/validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
        )
        
        print(f"\n Información sobre el Dataset:")
        print(f"   - Train: {len(X_train_split)} ejemplos")
        print(f"   - Validation: {len(X_val_split)} ejemplos")
        print(f"   - Clases únicas: {len(set(y_train))}")
        
        # Encode sentences to vectors
        print("\nGenerando embeddings con spaCy...")
        X_train_vectors = self.encode_sentences(X_train_split)
        X_val_vectors = self.encode_sentences(X_val_split)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train_split)
        y_val_encoded = self.label_encoder.transform(y_val_split)
        
        # Train SVM
        print("\nEntrenando SVM...")
        self.clf.fit(X_train_vectors, y_train_encoded)
        self.is_trained = True
        
        # Evaluate on train and validation
        train_pred = self.clf.predict(X_train_vectors)
        val_pred = self.clf.predict(X_val_vectors)
        
        train_acc = accuracy_score(y_train_encoded, train_pred)
        val_acc = accuracy_score(y_val_encoded, val_pred)
        val_f1 = f1_score(y_val_encoded, val_pred, average='macro')
        
        results = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1,
            "n_train": len(X_train_split),
            "n_val": len(X_val_split),
            "n_classes": len(self.label_encoder.classes_)
        }
        
        print("\n" + "="*60)
        print("✅ RESULTADOS DE ENTRENAMIENTO")
        print("="*60)
        print(f"   Train Accuracy:      {train_acc:.3f}")
        print(f"   Validation Accuracy: {val_acc:.3f}")
        print(f"   Validation F1-macro: {val_f1:.3f}")
        print("="*60)
        
        return results
    
    def predict(self, questions: List[str]) -> List[str]:
        """
        Predice intents para una lista de preguntas.
        
        Args:
            questions: Lista de preguntas
            
        Returns:
            Lista de intents predichos
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a train() primero.")
        
        X_vectors = self.encode_sentences(questions)
        y_encoded = self.clf.predict(X_vectors)
        y_pred = self.label_encoder.inverse_transform(y_encoded)
        
        return y_pred.tolist()
    
    def predict_single(self, question: str) -> str:
        """
        Predice el intent de una sola pregunta.
        
        Args:
            question: Pregunta a clasificar
            
        Returns:
            Intent predicho
        """
        return self.predict([question])[0]
    
    def evaluate(self, X_test: List[str], y_test: List[str], 
                 output_dir: str = "results") -> dict:
        """
        Evalúa el clasificador en un conjunto de test.
        
        Args:
            X_test: Lista de preguntas de test
            y_test: Lista de labels de test
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con métricas de evaluación
        """
        print("\n" + "="*60)
        print("🧪 EVALUACIÓN EN TEST SET")
        print("="*60)
        
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "n_test": len(X_test),
            "predictions": y_pred,
            "true_labels": y_test
        }
        
        print(f"\n📊 Métricas:")
        print(f"   Accuracy:  {acc:.3f}")
        print(f"   F1-macro:  {f1_macro:.3f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions to CSV
        df_results = pd.DataFrame({
            "question": X_test,
            "true_intent": y_test,
            "predicted_intent": y_pred,
            "is_correct": [t == p for t, p in zip(y_test, y_pred)]
        })
        df_results.to_csv(f"{output_dir}/svm_predictions.csv", index=False)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, save_path=f"{output_dir}/svm_confusion_matrix.png")
        
        print(f"\n💾 Resultados guardados en {output_dir}/")
        
        return results
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            save_path: str = "results/svm_cm.png"):
        """Visualiza matriz de confusión."""
        labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=labels, 
            yticklabels=labels,
            cbar_kws={'shrink': 0.8}
        )
        plt.title("SVM Intent Classification - Confusion Matrix", fontsize=14)
        plt.ylabel("True Intent")
        plt.xlabel("Predicted Intent")
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Matriz de confusión guardada en {save_path}")
        plt.close()
    
    def save_model(self, path: str = "models/svm_intent_classifier.pkl"):
        """Guarda el modelo entrenado."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            "clf": self.clf,
            "label_encoder": self.label_encoder,
            "is_trained": self.is_trained,
            "vector_dim": self.vector_dim
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo guardado en {path}")
    
    def load_model(self, path: str = "models/svm_intent_classifier.pkl"):
        """Carga un modelo previamente entrenado."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.clf = model_data["clf"]
        self.label_encoder = model_data["label_encoder"]
        self.is_trained = model_data["is_trained"]
        self.vector_dim = model_data["vector_dim"]
        
        print(f"✅ Modelo cargado desde {path}")
    
    @classmethod
    def load_test_set(cls) -> Tuple[List[str], List[str]]:
        """Carga el test set por defecto."""
        df = pd.DataFrame(cls.TEST_SET, columns=["question", "intent"])
        return df["question"].tolist(), df["intent"].tolist()


def main():
    """Ejemplo de uso del clasificador SVM."""
    
    # Inicializar clasificador
    classifier = SVMIntentClassifier(spacy_model="en_core_web_md", C=1.0)
    
    # Cargar datos de entrenamiento (usar el test set como ejemplo)
    # En un caso real, deberías tener un dataset de entrenamiento más grande
    X_train, y_train = classifier.load_test_set()
    
    print(f"\n⚠️ NOTA: Este es un ejemplo con datos limitados.")
    print(f"Para mejores resultados, usa un dataset más grande (ej: ATIS, Jarbas/core_intents)")
    
    # Entrenar
    train_results = classifier.train(X_train, y_train, validation_split=0.3)
    
    # Evaluar en el conjunto completo (solo para demostración)
    X_test, y_test = classifier.load_test_set()
    test_results = classifier.evaluate(X_test, y_test, output_dir="results/svm")
    
    # Guardar modelo
    classifier.save_model("models/svm_intent_classifier.pkl")
    
    # Ejemplo de predicción individual
    print("\n" + "="*60)
    print("🔮 EJEMPLOS DE PREDICCIÓN")
    print("="*60)
    examples = [
        "How do I reset my device?",
        "What is the maximum power consumption?",
        "The device is not turning on",
        "Is it safe to use outdoors?"
    ]
    
    for question in examples:
        intent = classifier.predict_single(question)
        print(f"❓ '{question}'")
        print(f"   → Intent: {intent}\n")


if __name__ == "__main__":
    main()