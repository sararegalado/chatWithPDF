import numpy as np
import pandas as pd
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from transformers import (
    AutoTokenizer, 
    AutoModel,
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')


class ContextualEmbeddings:
    def __init__(self, df_processed: pd.DataFrame, device: Optional[str] = None):
        self.df = df_processed
        
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Usando dispositivo: {self.device}")
        
        # Modelos
        self.bert_model = None
        self.bert_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.sentence_transformer = None
        
        # Embeddings generados
        self.bert_embeddings = None
        self.roberta_embeddings = None
        self.sbert_embeddings = None
        
        # Análisis
        self.similarity_analysis = {}
    
    
    # Funcion para cargar modelo BERT
    def load_bert(self, model_name: str = 'bert-base-uncased'):
        print(f"\nCargando BERT: {model_name}")
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        print("BERT cargado correctamente")
    
    
    # Funcion para crear embeddings BERT
    def create_bert_embeddings(self, pooling: str = 'mean', batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        if self.bert_model is None:
            raise ValueError("Carga BERT primero con load_bert()")
        
        print(f"\nGenerando embeddings BERT (pooling={pooling})")
        
        # Preparar textos (no se usa lemmas para mejorar el contexto)
        texts = self.df['clean_text'].tolist()
        
        embeddings = []
        
        # Procesar en batches con barra de progreso
        for i in tqdm(range(0, len(texts), batch_size), desc="Procesando batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar
            encoded = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Mover a dispositivo
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Aplicar pooling
            last_hidden = outputs.last_hidden_state 
            
            if pooling == 'cls':
                # Usar token [CLS]
                batch_embeddings = last_hidden[:, 0, :].cpu().numpy()
            
            elif pooling == 'mean':
                # Promedio ponderado por attention mask
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = (last_hidden * mask).sum(1)
                sum_mask = mask.sum(1).clamp(min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            elif pooling == 'max':
                # Max pooling
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).bool()
                last_hidden_masked = last_hidden.masked_fill(~mask, -1e9)
                batch_embeddings = last_hidden_masked.max(1).values.cpu().numpy()
            
            else:
                raise ValueError("no se ha podido aplicar pooling")
            
            embeddings.append(batch_embeddings)
        
        # Concatenar todos los batches
        self.bert_embeddings = np.vstack(embeddings)
        
        print(f"Shape de embeddings BERT: {self.bert_embeddings.shape}")
        return self.bert_embeddings
    
    
    # Funcion para cargar modelo ROBERTA (mejor que BERT en muchos casos)
    def load_roberta(self, model_name: str = 'roberta-base'):
        print(f"\nCargando RoBERTa: {model_name}")
        
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta_model = RobertaModel.from_pretrained(model_name)
        self.roberta_model.to(self.device)
        self.roberta_model.eval()
        
        print("RoBERTa cargado correctamente")
    
    
    def create_roberta_embeddings(self, pooling: str = 'mean', batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        if self.roberta_model is None:
            raise ValueError("Carga RoBERTa primero con load_roberta()")
        
        print(f"\nGenerando embeddings RoBERTa (pooling={pooling})")
        
        texts = self.df['clean_text'].tolist()
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Procesando batches"):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.roberta_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.roberta_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            last_hidden = outputs.last_hidden_state
            
            if pooling == 'cls':
                batch_embeddings = last_hidden[:, 0, :].cpu().numpy()
            elif pooling == 'mean':
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = (last_hidden * mask).sum(1)
                sum_mask = mask.sum(1).clamp(min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            elif pooling == 'max':
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).bool()
                last_hidden_masked = last_hidden.masked_fill(~mask, -1e9)
                batch_embeddings = last_hidden_masked.max(1).values.cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        self.roberta_embeddings = np.vstack(embeddings)
        
        print(f"Shape de embeddings RoBERTa: {self.roberta_embeddings.shape}")
        return self.roberta_embeddings
    

    
    def load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"\nCargando Sentence-Transformer: {model_name}")
        self.sentence_transformer = SentenceTransformer(model_name)
        
        
        print(f"Sentence-Transformer cargado (dim={self.sentence_transformer.get_sentence_embedding_dimension()})")
    
    
    def create_sbert_embeddings(self, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        if self.sentence_transformer is None:
            raise ValueError("No cargado ningun modelo Sentence-Transformer")
        
        print("\nGenerando embeddings Sentence-BERT")
        
        texts = self.df['clean_text'].tolist()
        
        self.sbert_embeddings = self.sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para cosine similarity
        )
        
        print(f"Shape de embeddings S-BERT: {self.sbert_embeddings.shape}")
        return self.sbert_embeddings
    
    
   
    # Funcion para comparar documentos similares encontrados por diferentes modelos  
    def compare_models_similarity(self, doc_idx: int = 0, top_k: int = 5):
        print(f"Comparación de modelos - Documento: {self.df.iloc[doc_idx]['filename']}")
        
        results = {}
        
        # BERT
        if self.bert_embeddings is not None:
            sim_matrix = cosine_similarity(self.bert_embeddings)
            similarities = sim_matrix[doc_idx]
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            print(f"\nBERT - Top {top_k} documentos similares:")
            for rank, idx in enumerate(top_indices, 1):
                print(f"  {rank}. {self.df.iloc[idx]['filename']} (sim: {similarities[idx]:.4f})")
            
            results['bert'] = [(self.df.iloc[idx]['filename'], similarities[idx]) 
                              for idx in top_indices]
        
        # RoBERTa
        if self.roberta_embeddings is not None:
            sim_matrix = cosine_similarity(self.roberta_embeddings)
            similarities = sim_matrix[doc_idx]
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            print(f"\nRoBERTa - Top {top_k} documentos similares:")
            for rank, idx in enumerate(top_indices, 1):
                print(f"  {rank}. {self.df.iloc[idx]['filename']} (sim: {similarities[idx]:.4f})")
            
            results['roberta'] = [(self.df.iloc[idx]['filename'], similarities[idx]) 
                                 for idx in top_indices]
        
        # Sentence-BERT
        if self.sbert_embeddings is not None:
            sim_matrix = cosine_similarity(self.sbert_embeddings)
            similarities = sim_matrix[doc_idx]
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            print(f"\nSentence-BERT - Top {top_k} documentos similares:")
            for rank, idx in enumerate(top_indices, 1):
                print(f"  {rank}. {self.df.iloc[idx]['filename']} (sim: {similarities[idx]:.4f})")
            
            results['sbert'] = [(self.df.iloc[idx]['filename'], similarities[idx]) 
                               for idx in top_indices]
        
        
        return results
    
    # Función para visualizar embeddings en 2D usando PCA o TSNE
    def visualize_embeddings(self, model_name: str = 'sbert', method: str = 'tsne', n_docs: Optional[int] = None, color_by: Optional[str] = None):
        if model_name == 'bert':
            embeddings = self.bert_embeddings
        elif model_name == 'roberta':
            embeddings = self.roberta_embeddings
        elif model_name == 'sbert':
            embeddings = self.sbert_embeddings
        else:
            raise ValueError("model_name debe ser 'bert', 'roberta' o 'sbert'")
        
        if embeddings is None:
            raise ValueError(f"No hay embeddings para {model_name}")
        
        # Limitar número de documentos si se especifica
        if n_docs is not None:
            embeddings = embeddings[:n_docs]
            df_subset = self.df.iloc[:n_docs]
        else:
            df_subset = self.df
        
        # Reducir dimensionalidad
        print(f"\nReduciendo dimensionalidad con {method.upper()}...")
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            variance = reducer.explained_variance_ratio_
            title = f'{model_name.upper()} Embeddings - PCA\n(Varianza explicada: {sum(variance)*100:.1f}%)'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords = reducer.fit_transform(embeddings)
            title = f'{model_name.upper()} Embeddings - t-SNE'
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        
        if color_by and color_by in df_subset.columns:
            # Colorear por categoría
            unique_vals = df_subset[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
            
            for i, val in enumerate(unique_vals):
                mask = df_subset[color_by] == val
                plt.scatter(coords[mask, 0], coords[mask, 1], 
                          c=[colors[i]], label=str(val), alpha=0.6, s=100)
            plt.legend()
        else:
            plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=100)
        
        # Anotar algunos documentos
        for i in range(min(10, len(df_subset))):
            filename = df_subset.iloc[i]['filename'][:15]  # Truncar nombre
            plt.annotate(filename, (coords[i, 0], coords[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.title(title, fontsize=14)
        plt.xlabel('Dimensión 1')
        plt.ylabel('Dimensión 2')
        plt.tight_layout()
        
        # Guardar
        output_path = f'data/visualizations/embeddings_{model_name}_{method}.png'
        os.makedirs('data/visualizations', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en {output_path}")
        plt.close()
    
    
    # Función para calcular estadísticas de los embeddings
    def compute_embedding_statistics(self) -> Dict:
        stats = {}
        
        for name, embeddings in [
            ('BERT', self.bert_embeddings),
            ('RoBERTa', self.roberta_embeddings),
            ('Sentence-BERT', self.sbert_embeddings)
        ]:
            if embeddings is not None:
                stats[name] = {
                    'shape': embeddings.shape,
                    'dimensionality': embeddings.shape[1],
                    'mean': float(embeddings.mean()),
                    'std': float(embeddings.std()),
                    'min': float(embeddings.min()),
                    'max': float(embeddings.max()),
                    'sparsity': float((embeddings == 0).sum() / embeddings.size * 100)
                }
        
        return stats
    
    

    # Funciones para guardar y cargar embeddings    
    def save_embeddings(self, output_dir: str = 'data/embeddings'):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.bert_embeddings is not None:
            np.save(f'{output_dir}/bert_embeddings.npy', self.bert_embeddings)
            print(f"BERT embeddings guardados")
        
        if self.roberta_embeddings is not None:
            np.save(f'{output_dir}/roberta_embeddings.npy', self.roberta_embeddings)
            print(f"RoBERTa embeddings guardados")
        
        if self.sbert_embeddings is not None:
            np.save(f'{output_dir}/sbert_embeddings.npy', self.sbert_embeddings)
            print(f"Sentence-BERT embeddings guardados")
        
        # Guardar estadísticas
        stats = self.compute_embedding_statistics()
        with open(f'{output_dir}/embedding_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Estadísticas guardadas")
    


if __name__ == "__main__":
    print("Generando embeddings contextuales...\n")
    
    # Cargar datos procesados
    df_processed = pd.read_pickle('data/processed/processed_documents.pkl')
    print(f"\n{len(df_processed)} documentos cargados")
    ctx_embeddings = ContextualEmbeddings(df_processed)
    
    # Generar embeddings con sentence trasformers y BERT
    print("GENERANDO EMBEDDINGS CON SENTENCE TRANSFORMERS")
    print("="*70)
    
    ctx_embeddings.load_sentence_transformer('all-MiniLM-L6-v2')
    sbert_emb = ctx_embeddings.create_sbert_embeddings(batch_size=32)
    
    print("Generando embeddings con BERT")
    
    ctx_embeddings.load_bert('bert-base-uncased')
    bert_emb = ctx_embeddings.create_bert_embeddings(pooling='mean', batch_size=8)
    
  
    print("Análisis de similitud")
    # Comparar modelos en documento 0
    ctx_embeddings.compare_models_similarity(doc_idx=0, top_k=5)
    
    # Estadísticas
    print("\nEstadísticas de embeddings:")
    stats = ctx_embeddings.compute_embedding_statistics()
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        for key, value in model_stats.items():
            if key != 'shape':
                print(f"  {key}: {value}")
    
    # Visualizaciones
    print("\nGenerando visualizaciones...")
    ctx_embeddings.visualize_embeddings('sbert', method='tsne')
    ctx_embeddings.visualize_embeddings('bert', method='pca')
    
    # Guardar embeddings
    print("\nGuardando embeddings...")
    ctx_embeddings.save_embeddings()
    # Guardar mapping: índice → filename
    embedding_index = {
        "filenames": df_processed['filename'].tolist(),
        "total_documents": len(df_processed)
    }
    with open('data/embeddings/embedding_index.json', 'w', encoding='utf-8') as f:
        json.dump(embedding_index, f, indent=2, ensure_ascii=False)
    print("Índice de embeddings guardado: data/embeddings/embedding_index.json")
