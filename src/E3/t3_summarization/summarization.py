import math
from typing import List, Optional
from pathlib import Path
import importlib.util

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm

from rouge_score import rouge_scorer
from bert_score import score as bertscore


#Cargamos la funcion de chunking de la tarea anterior 
def _load_smart_chunk_text():
    try:
        from src.E3.t2_retrieval.chunking import smart_chunk_text  # type: ignore
        return smart_chunk_text
    except Exception:
        # En caso de fallo localizar el chunking.py relativo al proyecto
        current = Path(__file__).resolve()
        src_root = current.parents[2]  # .../src
        chunking_path = src_root / "E3" / "t2_retrieval" / "chunking.py"
        if not chunking_path.exists():
            raise ImportError(f"chunking.py not found at {chunking_path}")

        spec = importlib.util.spec_from_file_location("chunking_fallback", str(chunking_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return getattr(module, "smart_chunk_text")


class Summarizer:
    """Abstractive summarizer en dos modos: 
    - 'single': resumir todo el texto completo en una sola vez
    - 'hierarchical': modo principal, dividir el texto en chunks usando `smart_chunk_text`, resumir cada chunk, y luego resumir los resúmenes concatenados (BART tiene límites de longitud)
    """
    #PREPARAZAR EL SISTEMA PARA RESUMEN ABSTRACCIVO USANDO BART
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # BART practical limits
        self.max_input_tokens = 1024
        self.smart_chunk_text = _load_smart_chunk_text()

    def _generate_summary_batch(self, texts: List[str], max_len: int = 150, num_beams: int = 4) -> List[str]:
        # Tokenize batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        #genererar los resumenes
        summary_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_len, #Controla la longitud del resumen
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3 #sin repetir frases 
        )

        summaries = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]
        return summaries

    def summarize(self, text: str, mode: str = "hierarchical", max_len: int = 150, max_len_final: int = 200) -> str:
        mode = mode.lower()
        if mode not in {"single", "hierarchical"}:
            raise ValueError("mode must be 'single' or 'hierarchical'")

        # usa todo el texto si cabe en 1024 tokens 
        if mode == "single":
            summaries = self._generate_summary_batch([text], max_len=max_len)
            return summaries[0] 
        
        # hierarchical: divide el texto en trozos (chunks)
        chunks = self.smart_chunk_text(text, max_chunk_size=400, overlap=50)
        if len(chunks) == 0:
            return ""
        if len(chunks) == 1:
            return self._generate_summary_batch([chunks[0]], max_len=max_len)[0]

        # Procesamos los chunks en batchs de 4 para eficiencia
        chunk_summaries = []
        batch_size = 4
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_summaries = self._generate_summary_batch(batch, max_len=max_len)
            chunk_summaries.extend(batch_summaries)

        # Combinamos los resumenes de los chunks y creamos el resumen final
        combined = "\n".join(chunk_summaries)
        final_summary = self._generate_summary_batch([combined], max_length=max_len_final)[0]
        return final_summary

    # Evaluación con embeddings de ROBERTa
    def evaluate(self, preds: List[str], refs: List[str], lang: str = "en") -> dict:
        if not preds or not refs or len(preds) != len(refs):
            raise ValueError("Preds and refs must be non-empty lists of the same length")

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        rouge1, rouge2, rougel = [], [], []
        for p, r in zip(preds, refs):
            scores = scorer.score(r, p)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougel.append(scores["rougeL"].fmeasure)

        P, R, F1 = bertscore(preds, refs, lang=lang)

        return {
            "ROUGE-1": float(sum(rouge1) / len(rouge1)),
            "ROUGE-2": float(sum(rouge2) / len(rouge2)),
            "ROUGE-L": float(sum(rougel) / len(rougel)),
            "BERTScore": float(F1.mean())
        }


if __name__ == "__main__":
    summarizer = Summarizer()

    try:
        text = open("sample_manual.txt", encoding="utf-8").read()
    except FileNotFoundError:
        text = (
            "This device manual explains installation, operation and safety. "
            "Follow the steps to install, configure, and maintain the device. "
            "Always disconnect power before servicing and wear protective gloves when required."
        )

    print("\n--- Generating summary (hierarchical) ---")
    summary = summarizer.summarize(text, mode="hierarchical", max_len=60, max_len_final=120)
    print(summary)

    gold_summary = "The manual explains installation, operation and key safety procedures."
    metrics = summarizer.evaluate([summary], [gold_summary])
    print("\n--- EVALUATION METRICS ---")
    print(metrics)
