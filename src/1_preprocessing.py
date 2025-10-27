import os
import re
from pdfminer.high_level import extract_text
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
import json
from typing import List, Dict
import gdown
import zipfile
import os

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cargar modelo spaCy para inglés para lemmatizer y pos tags
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 3000000

# Clase para el preprocesamiento
class DocumentPreprocessor:

    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.raw_documents = []
        self.processed_documents = []
        self.statistics = {
            'total_documents': 0,
            'total_tokens': 0,
            'total_sentences': 0,
            'unique_tokens': 0,
            'avg_doc_length': 0
        }
    
    # Funcion para extraer texto usando pdfminer
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            #laparams = LAParams(
                #line_margin=0.5,
                #word_margin=0.1,
               # boxes_flow=0.5
            #)
            text = extract_text(pdf_path)
            return text
        except Exception as e:
            print(f"Error extrayendo {pdf_path}: {e}")
            return ""
    
    # Funcion para limpiar texto
    def clean_text(self, text: str) -> str:
        
        # Eliminar múltiples espacios y saltos de línea
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres de control
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)        
        # Eliminar números de página comunes
        text = re.sub(r'(Página|Page)\s*\d+', '', text, flags=re.IGNORECASE)
        
        # Normalizar espacios
        text = ' '.join(text.split())
        
        return text.strip()
    
    # Funcion para tokenizar
    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text, language='english')
        return tokens
    
    # Funcion para case folding
    def case_folding(self, tokens: List[str]) -> List[str]:
        return [token.lower() for token in tokens]
    
    # Funcion para eliminar puntuacion
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.isalnum()]
    
    # Eliminar stopwords
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]
    
    # Funcion para aplicar stemming
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]
    
    # Funcion para aplicar lematizacion
    def lemmatize_text(self, text: str) -> List[str]:
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        return lemmas
    
    # Obterner etiquetas POS
    def get_pos_tags(self, text: str) -> List[tuple]:
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    # Funcion para procesar un documento completo con todas las tecnicas
    def process_document(self, text: str) -> Dict:
      
        clean_text = self.clean_text(text)
        
        tokens = self.tokenize(clean_text)
        
        tokens_lower = self.case_folding(tokens)
        
        tokens_clean = self.remove_punctuation(tokens_lower)
        
        tokens_no_stops = self.remove_stopwords(tokens_clean)
        
        tokens_stemmed = self.stem_tokens(tokens_no_stops)
        
        lemmas = self.lemmatize_text(clean_text)
        lemmas_lower = [lemma.lower() for lemma in lemmas]
        lemmas_clean = [lemma for lemma in lemmas_lower if lemma.isalnum()]
        lemmas_no_stops = [lemma for lemma in lemmas_clean if lemma not in self.stop_words]
  
        
        sentences = sent_tokenize(clean_text, language='english')
        
        # POS tagging
        pos_tags = self.get_pos_tags(clean_text)
        
        return {
            'raw_text': text,
            'clean_text': clean_text,
            'tokens_original': tokens,
            'tokens_processed': tokens_no_stops,
            'tokens_stemmed': tokens_stemmed,
            'lemmas': lemmas_no_stops,
            'sentences': sentences,
            'pos_tags': pos_tags,
            'num_tokens': len(tokens_no_stops),
            'num_sentences': len(sentences)
        }
    

    # Funcion para procesar todos los documentos
    def process_all_pdfs(self) -> pd.DataFrame:
        results = []
        print("\nPROCESANDO DOCUMENTOS PDF\n")
        
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        for idx, filename in enumerate(pdf_files, 1):
            pdf_path = os.path.join(self.pdf_directory, filename)
            print(f"\n[{idx}/{len(pdf_files)}] {filename}")
            
            # Extraer texto
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text:
                print(f"No se pudo extraer texto")
                continue
            
            # Procesar documento
            processed = self.process_document(text)
            
            # Imprimir estadisticas
            print(f"Tokens: {processed['num_tokens']}")
            print(f"Oraciones: {processed['num_sentences']}")
            
            # Solo lo guardamos si tiene al menos 1 frase
            if processed['num_tokens'] > 0:
                results.append({
                    'filename': filename,
                    'raw_text': processed['raw_text'],
                    'clean_text': processed['clean_text'],
                    'tokens': processed['tokens_processed'],
                    'tokens_stemmed': processed['tokens_stemmed'],
                    'lemmas': processed['lemmas'],
                    'sentences': processed['sentences'],
                    'num_tokens': processed['num_tokens'],
                    'num_sentences': processed['num_sentences']
                })
            else:
                print(f"No se ha encontrado texto en el archivo {filename}")
        
        df = pd.DataFrame(results)
        
        # Calcular estadísticas globales
        self.statistics['total_documents'] = len(df)
        self.statistics['total_tokens'] = df['num_tokens'].sum()
        self.statistics['total_sentences'] = df['num_sentences'].sum()
        
        all_tokens = [token for tokens in df['tokens'] for token in tokens]
        self.statistics['unique_tokens'] = len(set(all_tokens))
        self.statistics['avg_doc_length'] = df['num_tokens'].mean()
        
        return df
    
    # Funcion para guardar todos los datos procesados
    def save_processed_data(self, df: pd.DataFrame, output_dir: str = 'data/processed'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar DataFrame como pickle para preservar listas
        df.to_pickle(f'{output_dir}/processed_documents.pkl')
        print(f"Datos guardados correctamente")
        
        # Guardar estadísticas
        with open(f'{output_dir}/statistics.json', 'w', encoding='utf-8') as f:
            stats_json = {
                'total_documents': int(self.statistics['total_documents']),  
                'total_tokens': int(self.statistics['total_tokens']),
                'total_sentences': int(self.statistics['total_sentences']),
                'unique_tokens': int(self.statistics['unique_tokens']),
                'avg_doc_length': float(self.statistics['avg_doc_length'])
            }
            json.dump(stats_json, f, indent=2, ensure_ascii=False)
        print(f"Estadísticas guardadas")



# Main
if __name__ == "__main__":
    # Descargar archivos
    url = "https://drive.google.com/uc?id=19LS8c-asV9nRt2cF_0ccGnmKs9Gm6HQI"
    output = "raw.zip"
    folder = "data/"

    if not os.path.exists("data/raw/"):
        print("Descargando pdfs desde Google Drive...")
        gdown.download(url, output, quiet=False)

        print("Descomprimiendo datos...")
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(folder)

        os.remove(output)
        print("Datos listos en ./data/raw")
    else:
        print("Los datos ya existen en ./data/")


    # Crear preprocessor
    preprocessor = DocumentPreprocessor('data/raw/')
    
    # Procesar todos los PDFs
    df_processed = preprocessor.process_all_pdfs()
    
    #Guardar resultados
    preprocessor.save_processed_data(df_processed)
    
    # Mostrar ejemplo de documento procesado
    print("\nEJEMPLO DE DOCUMENTO PROCESADO\n")
    doc = df_processed.iloc[0]
    print(f"Archivo: {doc['filename']}")
    print(f"Texto sin procesar: {doc['raw_text']}")
    print(f"Texto procesado: {doc['clean_text']}")
