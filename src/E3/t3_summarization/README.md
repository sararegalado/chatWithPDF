# T3: Sumarización Abstractiva

## Descripción

Este módulo implementa sumarización abstractiva de textos largos (manuales de instrucciones) usando el modelo **BART** preentrenado de Facebook.

**Características:**
- Dos modos de sumarización: `single` (truncado simple) y `hierarchical` (chunking + resumen jerárquico)
- Integración con `smart_chunk_text` del módulo de retrieval (T2)
- Generación por lotes para eficiencia
- Evaluación con métricas ROUGE-1/2/L y BERTScore

## Instalación

Asegúrate de tener las dependencias instaladas:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers rouge-score bert-score tqdm
```

## Uso

### Modo Simple (Single)

```python
from src.E3.t3_summarization.summarization import Summarizer

summarizer = Summarizer()
text = "Your long text here..."
summary = summarizer.summarize(text, mode='single', max_len=150)
print(summary)
```

### Modo Jerárquico (Hierarchical)

Recomendado para textos muy largos (>1000 palabras). El summarizador:
1. Divide el texto en chunks usando `smart_chunk_text`
2. Resume cada chunk en paralelo
3. Concatena los resúmenes
4. Genera el resumen final

```python
from src.E3.t3_summarization.summarization import Summarizer

summarizer = Summarizer()
text = "Very long manual text..."
summary = summarizer.summarize(
    text, 
    mode='hierarchical', 
    max_len=120,        # Longitud máxima de cada resumen de chunk
    max_len_final=200   # Longitud del resumen final
)
print(summary)
```

### Evaluación

Evalúa la calidad del resumen comparándolo con resúmenes de referencia:

```python
predictions = ["Generated summary 1", "Generated summary 2"]
references = ["Reference summary 1", "Reference summary 2"]

metrics = summarizer.evaluate(predictions, references, lang='en')
print(metrics)
# Salida: {'ROUGE-1': 0.45, 'ROUGE-2': 0.28, 'ROUGE-L': 0.42, 'BERTScore': 0.87}
```

## API Completa

### Clase `Summarizer`

#### `__init__(model_name='facebook/bart-large-cnn', device=None)`
- **model_name**: Modelo BART a usar (por defecto: preentrenado en CNN)
- **device**: 'cpu', 'cuda' o None (auto-detecta)

#### `summarize(text, mode='hierarchical', max_len=150, max_len_final=200)`
- **text** (str): Texto a resumir
- **mode** (str): 'single' o 'hierarchical'
- **max_len** (int): Longitud máxima en tokens del resumen de chunks
- **max_len_final** (int): Longitud máxima en tokens del resumen final
- **return**: Resumen generado (str)

#### `evaluate(preds, refs, lang='en')`
- **preds** (List[str]): Resúmenes generados
- **refs** (List[str]): Resúmenes de referencia
- **lang** (str): Idioma para BERTScore
- **return**: Dict con métricas ROUGE-1/2/L y BERTScore

## Métricas

- **ROUGE-1**: Overlap de unigramas (palabras individuales)
- **ROUGE-2**: Overlap de bigramas (pares de palabras)
- **ROUGE-L**: Longest Common Subsequence (secuencia más larga común)
- **BERTScore**: Similitud semántica basada en embeddings de BERT

## Parámetros Recomendados

**Para resúmenes cortos (extractivo-like):**
```python
summarizer.summarize(text, mode='single', max_len=80)
```

**Para resúmenes balanceados:**
```python
summarizer.summarize(text, mode='hierarchical', max_len=120, max_len_final=150)
```

**Para resúmenes detallados:**
```python
summarizer.summarize(text, mode='hierarchical', max_len=150, max_len_final=250)
```

## Ejemplo Completo

```python
from src.E3.t3_summarization.summarization import Summarizer

# Inicializar
summarizer = Summarizer()

# Texto de ejemplo (manual simplificado)
manual_text = """
Installation Guide.
Step 1: Unpack the device from the box.
Step 2: Connect the power cable.
Step 3: Press the power button.
Safety Warnings:
- Do not operate with wet hands.
- Ensure proper ventilation.
- Contact support if malfunctioning.
Maintenance:
- Clean exterior with dry cloth.
- Check for loose parts monthly.
"""

# Generar resumen
summary = summarizer.summarize(manual_text, mode='hierarchical', max_len=60, max_len_final=100)
print("Resumen:")
print(summary)

# Evaluar contra referencia
gold_summary = "The manual provides installation steps, safety warnings, and maintenance instructions."
metrics = summarizer.evaluate([summary], [gold_summary])
print("\nMétricas:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## Notas Técnicas

- **Modelo por defecto**: `facebook/bart-large-cnn` (1.63 GB, optimizado para resumen en CNN Daily Mail)
- **Chunking**: Usa `smart_chunk_text` de `src/E3/t2_retrieval/chunking.py` (máx 400 palabras por chunk, overlap de 50)
- **Generación**: Beam search (4 beams), sin repetición de n-gramas
- **Tiempo**: ~25 seg por 1 modelo BART descargado en primera ejecución (CPU-only)

## Troubleshooting

**Error: "ModuleNotFoundError: Could not import module 'BartForConditionalGeneration'"**
- Solución: Reinstala PyTorch desde el índice oficial:
  ```bash
  pip uninstall -y torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

**Error: "chunking.py not found"**
- Asegúrate de ejecutar desde la raíz del repositorio o con PYTHONPATH configurado correctamente.

**Rendimiento lento**
- CPU vs GPU: Para GPU NVIDIA, instala `torch` con CUDA (ver https://pytorch.org/get-started/locally/)
- Modelos más ligeros: Prueba `sshleifer/distilbart-cnn-6-6` (descarga más rápida, ligeramente menos preciso)

## Referencias

- **BART Paper**: [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- **ROUGE**: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- **BERTScore**: [BERTScore: Evaluating Text Generation with BERT](https://openreview.net/pdf?id=SkeHuCVFDr)
