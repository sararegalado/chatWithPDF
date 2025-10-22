import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import os
import json
import gensim.downloader as api

class WordEmbeddings:    
    def __init__(self, df_processed: pd.DataFrame):
        self.df = df_processed
        self.w2v_model = None
        self.fasttext_model = None
        self.glove_vectors = None
        self.vocabulary = None
        self.oov_analysis = {}
    
    # Funcion que prepara las oraciones
    def prepare_sentences(self) -> List[List[str]]:
        all_sentences = []
        for doc_lemmas in self.df['lemmas']:
            chunk_size = 10
            for i in range(0, len(doc_lemmas), chunk_size):
                sentence = doc_lemmas[i:i+chunk_size]
                if len(sentence) > 2: 
                    all_sentences.append(sentence)
        return all_sentences
    
    
    # Funcion para entrenar modelo Word2Vec custom
    def train_word2vec(self, vector_size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4, epochs: int = 10) -> Word2Vec:
        
        print("Entrenando modelo Word2Vec")
        
        sentences = self.prepare_sentences()
        print(f"Total de oraciones: {len(sentences)}")
        
        # Entrenar Word2Vec
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=1
        )
        
        vocab_size = len(self.w2v_model.wv)
        print(f"Tamaño del vocabulario modelo Word2Vec: {vocab_size}")
        
        return self.w2v_model
    
    # Funcion para entrenar modelo FastText custom
    def train_fasttext(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 10) -> FastText:
        print("Entrenando FastText")
        
        sentences = self.prepare_sentences()
        
        # Entrenar FastText
        self.fasttext_model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            word_ngrams=1 
        )
        
        vocab_size = len(self.fasttext_model.wv)
        print(f"Tamaño del vocabulario FastText: {vocab_size}")
        
        return self.fasttext_model
    
    # Cargar modelo GloVe pre-entrenado
    def load_pretrained_glove(self, model: str):
        print("Cargando GloVe pre-entrenado")
        
        try:
            # Cargar en formato de Gensim
            self.glove_vectors = api.load(model)
            print(f"GloVe cargado")
            print(f"Vocabulario: {len(self.glove_vectors)}")
        except Exception as e:
            print(f"Error cargando GloVe: {e}")
    
    # Funcion para analizar palabras fuera del vocabulario
    def analyze_oov_words(self, model_name: str = 'word2vec') -> Dict:
        print(f"Análisis palabras OOV (out of vocabulary)- {model_name.upper()}")
        
        # Seleccionar modelo
        if model_name == 'word2vec':
            model = self.w2v_model.wv if self.w2v_model else None
        elif model_name == 'fasttext':
            model = self.fasttext_model.wv if self.fasttext_model else None
        elif model_name == 'glove':
            model = self.glove_vectors
        else:
            raise ValueError("Modelo no válido")
        
        if model is None:
            print("Modelo no entrenado/cargado")
            return {}
        
        # Obtener todas las palabras únicas del corpus
        all_words = set()
        for lemmas in self.df['lemmas']:
            all_words.update(lemmas)
        
        # Identificar OOV
        oov_words = []
        in_vocab_words = []
        
        for word in all_words:
            if word in model:
                in_vocab_words.append(word)
            else:
                oov_words.append(word)
        
        # Estadísticas
        total_words = len(all_words)
        oov_count = len(oov_words)
        oov_percentage = (oov_count / total_words) * 100
        
        print(f"Total palabras únicas: {total_words}")
        print(f"Palabras en vocabulario: {len(in_vocab_words)}")
        print(f"Palabras OOV: {oov_count}")
        print(f"Porcentaje OOV: {oov_percentage:.2f}%")
        
        # Ejemplos de OOV
        print(f"\\nEjemplos de palabras OOV (primeras 10):")
        for word in oov_words[:10]:
            print(f"  - {word}")
        
        # Guardar análisis
        self.oov_analysis[model_name] = {
            'total_words': total_words,
            'in_vocab': len(in_vocab_words),
            'oov_count': oov_count,
            'oov_percentage': oov_percentage,
            # Guardar primeras 50 palabras OOV
            'oov_words': oov_words[:50]
        }
        
        return self.oov_analysis[model_name]
    
    
    # Funcion para calcular similitud entre dos palabras
    def get_word_similarity(self, word1: str, word2: str, model_name: str = 'word2vec') -> float:
        if model_name == 'word2vec':
            model = self.w2v_model.wv
        elif model_name == 'fasttext':
            model = self.fasttext_model.wv
        elif model_name == 'glove':
            model = self.glove_vectors
        
        try:
            similarity = model.similarity(word1, word2)
            return similarity
        except KeyError:
            return None
    
    
    # Funcion para encontrar palabras similares
    def find_similar_words(self, word: str, topn: int = 10, model_name: str = 'word2vec') -> List[Tuple[str, float]]:
        if model_name == 'word2vec':
            model = self.w2v_model.wv
        elif model_name == 'fasttext':
            model = self.fasttext_model.wv
        elif model_name == 'glove':
            model = self.glove_vectors
        
        try:
            similar = model.most_similar(word, topn=topn)
            return similar
        except KeyError:
            print(f"Palabra '{word}' no encontrada en vocabulario")
            return []
    
    
    # Funcion para visualizar embeddings con PCA o TSNE
    def visualize_embeddings(self, words: List[str] = None, n_words: int = 50, model_name: str = 'word2vec', method: str = 'tsne'):
        if model_name == 'word2vec':
            model = self.w2v_model.wv
        elif model_name == 'fasttext':
            model = self.fasttext_model.wv
        elif model_name == 'glove':
            model = self.glove_vectors
        
        # Seleccionar palabras
        if words is None:
            words = list(model.index_to_key[:n_words])
        
        # Obtener vectores
        vectors = np.array([model[word] for word in words if word in model])
        valid_words = [word for word in words if word in model]
        
        # Reducir dimensionalidad
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(vectors)
            title = f'Word Embeddings - PCA ({model_name})'
        else:
            reducer = TSNE(n_components=2, random_state=0)
            coords = reducer.fit_transform(vectors)
            title = f'Word Embeddings - t-SNE ({model_name})'
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
        
        # Anotar algunas palabras
        for i, word in enumerate(valid_words[:20]):
            plt.annotate(word, (coords[i, 0], coords[i, 1]))
        
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig(f'data/processed/visualizations_{model_name}_{method}.png', dpi=300, bbox_inches='tight')
        print(f"Visualización guardada")
        plt.close()
    
    
    # Función para convertir documentos completos en vectores promediando los embeddings de palabras individuales
    def create_document_embeddings(self, model_name: str = 'word2vec') -> np.ndarray:
        if model_name == 'word2vec':
            model = self.w2v_model.wv
        elif model_name == 'fasttext':
            model = self.fasttext_model.wv
        elif model_name == 'glove':
            model = self.glove_vectors
        
        doc_embeddings = []
        
        for lemmas in self.df['lemmas']:
            # Obtener vectores de palabras que están en vocabulario
            word_vectors = [model[word] for word in lemmas if word in model]
            
            if word_vectors:
                # Promedio de vectores
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Vector cero si no hay palabras en vocabulario
                doc_vector = np.zeros(model.vector_size)
            
            doc_embeddings.append(doc_vector)
        
        return np.array(doc_embeddings)
    
    # Funcion para guardar modelos entrenados
    def save_models(self, output_dir: str = 'models'):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.w2v_model:
            self.w2v_model.save(f'{output_dir}/word2vec.model')
            print(f"Word2Vec guardado")
        
        if self.fasttext_model:
            self.fasttext_model.save(f'{output_dir}/fasttext.model')
            print(f"FastText guardado")
        
        # Guardar análisis OOV
        with open(f'{output_dir}/oov_analysis.json', 'w', 
                 encoding='utf-8') as f:
            json.dump(self.oov_analysis, f, indent=2, ensure_ascii=False)
        print(f"Análisis OOV guardado")



if __name__ == "__main__":
    # Cargar datos procesados
    df_processed = pd.read_pickle('data/processed/processed_documents.pkl')
    
    # Crear clase
    embeddings_analyzer = WordEmbeddings(df_processed)
    
    # Entrenar Word2Vec
    embeddings_analyzer.train_word2vec(vector_size=100, epochs=10)
    
    # Entrenar FastText
    embeddings_analyzer.train_fasttext(vector_size=100, epochs=10)

    # Entrenar GloVe
    embeddings_analyzer.load_pretrained_glove('glove-wiki-gigaword-100')
    
    # Análisis OOV
    oov_w2v = embeddings_analyzer.analyze_oov_words('word2vec')
    oov_ft = embeddings_analyzer.analyze_oov_words('fasttext')
    oov_glove = embeddings_analyzer.analyze_oov_words('glove')
    
    # Comparar OOV entre modelos
    print("Comparación OOV de distintos modelos\n")
    print(f"Word2Vec OOV: {oov_w2v['oov_percentage']:.2f}%")
    print(f"FastText OOV: {oov_ft['oov_percentage']:.2f}%")
    print(f"GloVe OOV: {oov_glove['oov_percentage']:.2f}%")

    # Ejemplos de similitud
    print("\nEJEMPLOS DE SIMILITUD\n")
    test_word = "device"
    similar = embeddings_analyzer.find_similar_words(test_word, topn=5)
    print(f"Palabras similares a '{test_word}':")
    for word, score in similar:
        print(f"  {word}: {score:.3f}")
    
    # Visualizar embeddings
    embeddings_analyzer.visualize_embeddings(n_words=50, method='tsne')
    
    # Crear embeddings de documentos con modelo Word2Vec
    doc_embeddings = embeddings_analyzer.create_document_embeddings()
    np.save('data/processed/doc_embeddings_w2v.npy', doc_embeddings)
    
    # Guardar modelos
    embeddings_analyzer.save_models()