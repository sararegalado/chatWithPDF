# Chatbot para Interacción con Documentos PDF

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Un chatbot inteligente basado en NLP que permite interactuar con documentos PDF mediante preguntas en lenguaje natural, utilizando técnicas de Retrieval-Augmented Generation (RAG).

## 📋 Descripción

Este proyecto desarrolla un sistema de Question-Answering especializado en manuales de instrucciones. El chatbot es capaz de comprender preguntas en lenguaje natural y proporcionar respuestas contextualizadas basadas en el contenido de documentos PDF, mejorando significativamente la experiencia de búsqueda y consulta de información técnica.

### Motivación

Con la digitalización masiva de documentos, los manuales en formato PDF son cada vez más comunes. Aunque permiten búsquedas por palabras clave, no interpretan el contexto ni responden de manera inteligente. Este proyecto soluciona ese problema mediante tecnologías de NLP avanzadas.

## ✨ Características Principales

- 🔍 **Búsqueda Semántica**: Encuentra información relevante más allá de coincidencias exactas de palabras
- 💬 **Respuestas Contextualizadas**: Genera respuestas coherentes basadas en el contenido del documento
- 📄 **Resumen Automático**: Capacidad de resumir documentos PDF completos
- 🎯 **Fine-tuning Especializado**: Modelo optimizado para manuales de instrucciones
- 📊 **Trazabilidad**: Referencias a las secciones del documento de donde se extrae la información

## 🏗️ Arquitectura del Sistema

El sistema implementa un pipeline RAG (Retrieval-Augmented Generation) con las siguientes etapas:

1. **Ingesta y Preprocesamiento**
   - Extracción de texto de PDFs con `pdfminer`
   - Limpieza y normalización del texto
   - Segmentación en chunks significativos

2. **Indexación Vectorial**
   - Generación de embeddings semánticos
   - Construcción de índice vectorial para búsqueda eficiente
   - Similarity search optimizada

3. **Recuperación de Contexto (RAG)**
   - Information Retrieval de fragmentos relevantes
   - Question Answering contextualizado
   - Context grounding para respuestas precisas

4. **Generación de Respuestas**
   - LLM fine-tuneado para el dominio específico
   - Natural Language Generation (NLG)
   - Prompting engineering avanzado

## 📊 Dataset

El proyecto utiliza un dataset propio de **200-300 manuales de instrucciones** recopilados de [ManualsLib.com](https://www.manualslib.com/), cubriendo diversos productos y categorías.

### Estructura de Datos

```
data/
├── raw/              # PDFs originales de manuales
├── processed/        # Texto extraído y limpio
└── embeddings/       # Vectores generados
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip o conda

### Instalación Básica

```bash
# Clonar el repositorio
git clone https://github.com/usuario/chatbot-pdf-nlp.git
cd chatbot-pdf-nlp

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Instalación con Conda

```bash
conda env create -f environment.yml
conda activate chatbot-pdf
```

## 💻 Uso

### Ejemplo Básico

```python
from src.models.chatbot import PDFChatbot

# Inicializar el chatbot
chatbot = PDFChatbot()

# Cargar un manual
chatbot.load_pdf("data/raw/manual_ejemplo.pdf")

# Hacer una pregunta
respuesta = chatbot.ask("¿Cómo se cambia la batería?")
print(respuesta)
```

### Interfaz de Línea de Comandos

```bash
# Procesar un PDF
python scripts/process_pdf.py --input data/raw/manual.pdf

# Entrenar el modelo
python src/models/train.py --config configs/config.yaml

# Iniciar chatbot interactivo
python scripts/chat.py --pdf data/raw/manual.pdf
```

## 🧪 Tareas de NLP Implementadas

### Preprocesamiento
- Tokenización
- Normalización de texto
- Segmentación semántica

### Modelado
- Embeddings semánticos
- Information Retrieval
- Question Answering
- Natural Language Generation
- Abstractive Summarization (resúmenes)

### Fine-tuning
- Ajuste de LLM en dominio de manuales
- Prompting engineering
- Context grounding

## 📈 Evaluación

El sistema se evalúa mediante:

- **Precisión de Recuperación**: Métricas de información recuperada relevante
- **Calidad de Respuestas**: ROUGE, BLEU, BERT Score
- **Calidad de Resúmenes**: ROUGE-L, METEOR
- **Satisfacción del Usuario**: Encuestas post-interacción

## 🔬 Estado del Arte

Este proyecto se inspira en investigaciones y frameworks líderes:

### Frameworks
- **LangChain**: Orquestación de LLMs y RAG pipelines
- **Haystack**: Framework orientado a producción con retrievers y readers

### Aplicaciones Comerciales
- ChatPDF, ChatDOC, EaseMate

### Papers Relevantes
- "Towards Designing a Question-Answering Chatbot for Online News"
- "Understanding question-answering systems: Evolution, applications, trends"
- "Analysis of Language-Model-Powered Chatbots for Query Resolution in PDF-Based Automotive Manuals"

## 📁 Estructura del Proyecto

```
chatbot-pdf-nlp/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/              # Manuales PDF originales
│   ├── processed/        # Texto procesado
│   └── embeddings/       # Vectores e índices
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_preprocesamiento.ipynb
│   └── 03_modelado_rag.ipynb
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── pdf_extractor.py
│   ├── features/
│   │   ├── embeddings.py
│   │   └── chunking.py
│   ├── models/
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── chatbot.py
│   └── utils/
│       └── evaluation.py
├── models/               # Modelos entrenados
├── configs/
│   └── config.yaml
├── scripts/
│   ├── process_pdf.py
│   ├── train.py
│   └── chat.py
└── tests/
```

## 🛠️ Tecnologías Utilizadas

- **Extracción de PDFs**: pdfminer
- **Embeddings**: Sentence Transformers, OpenAI embeddings
- **Vector Store**: FAISS, Chroma
- **LLM**: GPT-4, LLaMA 2, o modelos similares
- **Framework RAG**: LangChain / Haystack
- **Evaluación**: ROUGE, BLEU, BERT Score

## 🎯 Roadmap

- [x] Extracción y preprocesamiento de PDFs
- [x] Sistema básico de RAG
- [ ] Fine-tuning del LLM
- [ ] Resumen automático de documentos
- [ ] Soporte multimodal (imágenes, tablas)
- [ ] Interfaz web con Streamlit/Gradio
- [ ] Sistema de feedback del usuario
- [ ] Despliegue en producción

## 👥 Autores

- **Zaloa Fernández Soto**
- **Sara Regalado Aramendi**

**Asignatura**: Fundamentos del Procesamiento del Lenguaje Natural  
**Fecha**: Septiembre 2025

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, contacta a los autores del proyecto.

## 🙏 Agradecimientos

- ManualsLib.com por proporcionar acceso a los manuales
- Comunidad de LangChain y Haystack
- Investigadores citados en el estado del arte

---

**Nota**: Este proyecto es parte de un trabajo académico para la asignatura de Fundamentos del Procesamiento del Lenguaje Natural.