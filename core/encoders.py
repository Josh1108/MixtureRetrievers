
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss

import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
)
from sentence_transformers import SentenceTransformer

MODEL_ALIASES: Dict[str, str] = {
    "simcse": "princeton-nlp/unsup-simcse-bert-base-uncased",
    "contriever": "facebook/contriever",
    "dpr": "facebook/dpr-ctx_encoder-single-nq-base",
    "ance": "castorini/ance-dpr-question-multi",
    "gtr-t5-base": "sentence-transformers/gtr-t5-base",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

DPR_CTX_DEFAULT = "facebook/dpr-ctx_encoder-single-nq-base"
DPR_QRY_DEFAULT = "facebook/dpr-question_encoder-single-nq-base"


def resolve_model_name(name: Optional[str]) -> str:
    if not name:
        return "sentence-transformers/all-mpnet-base-v2"
    low = name.strip().lower()
    return MODEL_ALIASES.get(low, name)


def is_dpr(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.lower()
    return ("dpr" in n) or ("facebook/dpr" in n) or ("facebook-dpr" in n)


def resolve_dpr_variant(name: Optional[str], is_query: bool) -> str:
    """If a DPR-ish name was provided, map to ctx/question concrete IDs."""
    if not name:
        return DPR_QRY_DEFAULT if is_query else DPR_CTX_DEFAULT
    low = name.lower().strip()
    # If caller already passed an explicit DPR ctx/question id, keep it
    if "question" in low:
        return name if is_query else DPR_CTX_DEFAULT
    if "ctx" in low or "context" in low:
        return DPR_QRY_DEFAULT if is_query else name
    # If alias like "dpr" or any DPR string
    if is_query:
        return DPR_QRY_DEFAULT
    return DPR_CTX_DEFAULT

class DPRTextEncoder:
    """HF DPR wrapper producing pooled vectors (ctx or question)."""
    def __init__(self, model_name: str, is_query: bool = False):
        tok_cls   = DPRQuestionEncoderTokenizerFast if is_query else DPRContextEncoderTokenizerFast
        model_cls = DPRQuestionEncoder               if is_query else DPRContextEncoder
        self.tok   = tok_cls.from_pretrained(model_name)
        self.model = model_cls.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                vec = self.model(**inputs, return_dict=True).pooler_output  # [B, H]
            outs.append(vec.detach().cpu().numpy().astype("float32"))
        embs = np.vstack(outs) if outs else np.zeros((0, 768), dtype="float32")
        faiss.normalize_L2(embs)
        return embs

class EmbedderCache:
    """
    Cache ST & DPR encoders.
    - SentenceTransformer for general cases
    - DPRTextEncoder for DPR, auto-selects ctx/question
    """
    def __init__(self, default_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.default_model = default_model
        self.st_models: Dict[str, SentenceTransformer] = {}
        self.dpr_models: Dict[Tuple[str, bool], DPRTextEncoder] = {}

    def _get_instance(self, name: Optional[str], is_query: bool):
        # Resolve aliases first
        base = resolve_model_name(name or self.default_model)
        if is_dpr(base):
            variant = resolve_dpr_variant(base, is_query=is_query)
            key = (variant, is_query)
            if key not in self.dpr_models:
                self.dpr_models[key] = DPRTextEncoder(variant, is_query=is_query)
            return self.dpr_models[key]
        # SentenceTransformer path
        if base not in self.st_models:
            self.st_models[base] = SentenceTransformer(base)
        return self.st_models[base]

    def encode(self, model_name: Optional[str], texts: List[str], batch_size: int = 64, is_query: bool = False) -> np.ndarray:
        m = self._get_instance(model_name, is_query=is_query)
        if isinstance(m, SentenceTransformer):
            out_parts = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                emb = m.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
                out_parts.append(emb.astype("float32", copy=False))
            embs = np.vstack(out_parts) if out_parts else np.zeros((0, 768), dtype="float32")
            faiss.normalize_L2(embs)
            return embs
        # DPR path
        return m.encode(texts, batch_size=batch_size)