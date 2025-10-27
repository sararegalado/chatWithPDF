from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import json
import os

# Clase para implementar Bag of Words y TF-IDF
class TraditionalRepresentations:
    
    def __init__(self, df_processed: pd.DataFrame):
        self.df = df_processed
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.bow_matrix = None
        self.tfidf_matrix = None
    
    # Preparar corpus juntando tokens
    def prepare_corpus(self) -> List[str]:
        series = self.df['lemmas']
        corpus = []

        for item in series:
            if isinstance(item, list):
                tokens = [str(t) for t in item if t is not None]
                corpus.append(' '.join(tokens))
            elif pd.isna(item):
                corpus.append('')
            else:
                corpus.append(str(item))

        return corpus
    
    # Bag of Words
    # Limitar vocabulario a 2000 palabras más frecuentes
    # Minimo debe aparecer minimo en 2 documentos
    # Palabra no debe aparecer en mas de 80% de documentos
    def create_bow_representation(self, max_features: int = 2000, min_df: int = 2, max_df: float = 0.8) -> np.ndarray:
        
        corpus = self.prepare_corpus()
        print("CREANDO REPRESENTACIÓN BAG OF WORDS")
        
        # Crear vectorizador
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'\b\w+\b'  # Palabras de 1+ caracteres
        )

        # Comprobar que el corpus no esta vacio
        if all((not doc or not doc.strip()) for doc in corpus):
            print("El corpus está vacío")
            # Matriz vacía (n_docs x 0)
            self.bow_matrix = np.zeros((len(corpus), 0))
            return self.bow_matrix

        # Crear matriz BoW
        try:
            self.bow_matrix = self.bow_vectorizer.fit_transform(corpus)
        except ValueError as e:
            print("Error al crear matriz BoW")
            self.bow_matrix = np.zeros((len(corpus), 0))
            return self.bow_matrix
            
        
        # Estadísticas
        vocab_size = len(self.bow_vectorizer.vocabulary_)
        density = (self.bow_matrix.nnz / 
                  (self.bow_matrix.shape[0] * self.bow_matrix.shape[1])) * 100
        
        print(f"Tamaño del vocabulario: {vocab_size}")
        print(f"Forma de la matriz: {self.bow_matrix.shape}")
        print(f"Densidad de la matriz: {density:.2f}%")
        print(f"Total de elementos no-cero: {self.bow_matrix.nnz}")
        
        return self.bow_matrix.toarray()
    
    # TF-IDF
    def create_tfidf_representation(self, max_features: int = 1000, min_df: int = 2, max_df: float = 0.8) -> np.ndarray:
        
        corpus = self.prepare_corpus()
        print("CREANDO REPRESENTACIÓN TF-IDF")
        
        # Crear vectorizador TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'\b\w+\b'
        )
        
        # Crear matriz TF-IDF
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Estadísticas
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        density = (self.tfidf_matrix.nnz / 
                  (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100
        
        print(f"Tamaño del vocabulario: {vocab_size}")
        print(f"Forma de la matriz: {self.tfidf_matrix.shape}")
        print(f"Densidad de la matriz: {density:.2f}%")
        
        return self.tfidf_matrix.toarray()
    
    # Obtener terminos mas importantes segun BoW o TF-IDF
    def get_top_terms(self, n_terms: int = 20, representation: str = 'tfidf') -> pd.DataFrame:
        
        if representation == 'bow':
            if self.bow_matrix is None:
                raise ValueError("No existe matriz BoW")
            matrix = self.bow_matrix.toarray()
            vectorizer = self.bow_vectorizer
        else:
            if self.tfidf_matrix is None:
                raise ValueError("No existe matriz TF-IDF")
            matrix = self.tfidf_matrix.toarray()
            vectorizer = self.tfidf_vectorizer
        
        # Sumar scores por término
        term_scores = matrix.sum(axis=0)
        
        # Obtener nombres de features
        feature_names = vectorizer.get_feature_names_out()
        
        # Crear DataFrame y ordenar
        terms_df = pd.DataFrame({
            'term': feature_names,
            'score': term_scores
        }).sort_values('score', ascending=False).head(n_terms)
        
        return terms_df
    
    # Calcular similitud cosenmo entre documentos
    def compute_document_similarity(self, representation: str = 'tfidf') -> np.ndarray:
        if representation == 'bow':
            matrix = self.bow_matrix
        else:
            matrix = self.tfidf_matrix
        
        similarity_matrix = cosine_similarity(matrix)
        return similarity_matrix
    
    # Funcion que encuentra documentos similares
    def find_similar_documents(self, doc_idx: int, n_similar: int = 5, representation: str = 'tfidf') -> pd.DataFrame:
        
        similarity_matrix = self.compute_document_similarity(representation)
        
        similarities = similarity_matrix[doc_idx]
        
        # Ordenar
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        results = []
        for idx in similar_indices:
            results.append({
                'filename': self.df.iloc[idx]['filename'],
                'similarity': similarities[idx]
            })
        
        return pd.DataFrame(results)
    
    
    # Funcion para crear visualizacion terminos mas frecuentes
    def visualize_top_terms(self, n_terms: int = 20):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # BoW
        bow_terms = self.get_top_terms(n_terms, 'bow')
        axes[0].barh(bow_terms['term'], bow_terms['score'])
        axes[0].set_xlabel('Frecuencia')
        axes[0].set_title('Top Términos - Bag of Words')
        axes[0].invert_yaxis()
        
        # TF-IDF
        tfidf_terms = self.get_top_terms(n_terms, 'tfidf')
        axes[1].barh(tfidf_terms['term'], tfidf_terms['score'])
        axes[1].set_xlabel('Score TF-IDF')
        axes[1].set_title('Top Términos - TF-IDF')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        os.makedirs('data/visualizations/', exist_ok=True)
        plt.savefig('data/visualizations/top_terms.png', dpi=300, bbox_inches='tight')
        print("Visualización guardada en data/visualizations/top_terms.png")
        plt.close()
    
    
    # Guardar matrices BoW y TF-IDF
    def save_representations(self, output_dir: str = 'data/visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar matrices
        if self.bow_matrix is not None:
            np.save(f'{output_dir}/bow_matrix.npy', 
                   self.bow_matrix.toarray())
            print(f"BoW guardado en {output_dir}/bow_matrix.npy")
        
        if self.tfidf_matrix is not None:
            np.save(f'{output_dir}/tfidf_matrix.npy', 
                   self.tfidf_matrix.toarray())
            print(f"TF-IDF guardado en {output_dir}/tfidf_matrix.npy")
        
        # Guardar vocabularios
        def _serialize_vocab(vocab):
            if vocab is None:
                return None
            return {str(k): int(v) for k, v in vocab.items()}

        vocab_data = {
            'bow_vocab': _serialize_vocab(self.bow_vectorizer.vocabulary_ if self.bow_vectorizer else None),
            'tfidf_vocab': _serialize_vocab(self.tfidf_vectorizer.vocabulary_ if self.tfidf_vectorizer else None)
        }
        os.makedirs('data/vocabularies/', exist_ok=True)
        with open(f'data/vocabularies/vocabularies.json', 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print("Vocabularios guardados")



# Main
if __name__ == "__main__":
    # Cargar datos procesados
    df_processed = pd.read_pickle('data/processed/processed_documents.pkl')

    # Crear representaciones tradicionales
    trad_repr = TraditionalRepresentations(df_processed)
    
    # Crear BoW
    bow_matrix = trad_repr.create_bow_representation(max_features=1000)
    
    # Crear TF-IDF
    tfidf_matrix = trad_repr.create_tfidf_representation(max_features=1000)
    
    # Analizar términos importantes
    print("\nTOP 10 TÉRMINOS MÁS IMPORTANTES (TF-IDF)")
    print(trad_repr.get_top_terms(10, 'tfidf'))
    
    # Encontrar documentos similares
    print("\nDOCUMENTOS SIMILARES AL PRIMERO")
    print(f"Documento de referencia: {df_processed.iloc[0]['filename']}")
    print(trad_repr.find_similar_documents(0, n_similar=3))
    
    # Visualizar
    trad_repr.visualize_top_terms(20)
    
    # Guardar
    trad_repr.save_representations()