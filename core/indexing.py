#!/usr/bin/env python3
"""
multi_retriever.py

Import and use:

from multi_retriever import MultiIndexRetriever

engine = MultiIndexRetriever(
    corpus_path="data/my_corpus",     # folder with *.jsonl or corpus/*.jsonl, each line: {"id": "...", "contents": "..."}
    output_dir="indexes",             # where to cache indexes + runs
    encoders="default",               # or list of encoder names (includes "bm25")
    index_type="flat",                # "flat" | "ivf" | "hnsw" for vector indexes
    batch_size=64,
    top_k=50,
    build_props=True                  # also build proposition corpus
)

# queries can be a path to .jsonl (each line: {"id": "...", "title": "..."}) or a list[dict]
engine.ensure_all_indexes()
files = engine.search_and_dump_4way(queries="data/queries.jsonl", save_snippets=True)
print(files)  # dict of 4 output paths
"""

import os
import re
import json
import time
import math
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_ENCODERS: List[str] = [
    "bm25",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-mpnet-base-dot-v1",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]

PROP_MODEL_NAME = "chentong00/propositionizer-wiki-flan-t5-large"

# -----------------------------
# Helpers
# -----------------------------
def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("multi_retriever")

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def _write_csv_triples(path: Path, rows: List[Tuple[str, str, float]], append: bool = False) -> None:
    """
    rows: list of (qid, doc_id, score), no header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for qid, did, score in rows:
            f.write(f"{qid},{did},{score:.6f}\n")


def _write_jsonl(path: Path, rows: List[dict], append: bool = False) -> None:
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _short_snippet(text: str, n: int = 240) -> str:
    t = (text or "").replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t

def _norm_l2(x: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(x)
    return x

def _ivf_nlist(n: int) -> int:
    return max(1, min(4096, int(math.sqrt(max(1, n)))))

# -----------------------------
# Core
# -----------------------------
class MultiIndexRetriever:
    def __init__(
        self,
        corpus_path: str,
        output_dir: str = "indexes",
        encoders: Optional[List[str]] = "default",
        index_type: str = "flat",
        batch_size: int = 64,
        top_k: int = 50,
        build_props: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or _setup_logger()
        self.corpus_path = Path(corpus_path)
        self.output_root = Path(output_dir)
        self.index_type = index_type
        self.batch_size = int(batch_size)
        self.top_k = int(top_k)
        self.build_props = build_props

        if self.corpus_path.is_dir():
            self.dataset_name = _sanitize(self.corpus_path.name)
        else:
            self.dataset_name = _sanitize(self.corpus_path.stem)
        self.dataset_dir = self.output_root / self.dataset_name
        _ensure_dir(self.dataset_dir)

        if encoders == "default" or encoders is None:
            self.encoders = DEFAULT_ENCODERS.copy()
        else:
            self.encoders = list(encoders)

        self._full_corpus: Optional[List[dict]] = None
        self._props_corpus: Optional[List[dict]] = None

    def search_and_dump_4way_csv_ids(
        self,
        queries: Any,                  # path or list[{"id","title"}]
        top_k: Optional[int] = None,
        run_tag_prefix: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Writes 4 * (#encoders) CSV files with rows 'qid,doc_id,score' (no header):
        <encoder>__query_chunk__<tag>.csv
        <encoder>__subquery_chunk__<tag>.csv
        <encoder>__query_prop__<tag>.csv
        <encoder>__subquery_prop__<tag>.csv
        Returns {"files": [paths...]}.
        """
        self.ensure_all_indexes()
        q_list = self._load_queries(queries)
        subqs  = self._prop_queries_same_model(q_list)

        tk  = top_k or self.top_k
        tag = run_tag_prefix or _now_tag()
        run_dir = self.dataset_dir / "runs"
        _ensure_dir(run_dir)

        queries_q    = [(q["id"], q["title"]) for q in q_list]
        queries_subq = [(s["id"], s["title"]) for s in subqs]

        written: List[str] = []

        for enc in self.encoders:
            enc_base = _sanitize(enc)
            f_qc = run_dir / f"{enc_base}__query_chunk__{tag}.csv"
            f_sc = run_dir / f"{enc_base}__subquery_chunk__{tag}.csv"
            f_qp = run_dir / f"{enc_base}__query_prop__{tag}.csv"
            f_sp = run_dir / f"{enc_base}__subquery_prop__{tag}.csv"

            # QUERY -> CHUNK
            res = self._search_bm25("chunk", queries_q, tk) if enc.lower()=="bm25" else self._search_vec(enc,"chunk",queries_q,tk)
            rows = []
            for qid, _ in queries_q:
                for rank, (doc_id, score) in enumerate(res.get(qid, []), start=1):
                    rows.append((qid, doc_id, score))
            if rows:
                _write_csv_triples(f_qc, rows, append=False); written.append(str(f_qc))

            # SUBQUERY -> CHUNK
            res = self._search_bm25("chunk", queries_subq, tk) if enc.lower()=="bm25" else self._search_vec(enc,"chunk",queries_subq,tk)
            rows = []
            for sid, _ in queries_subq:
                for rank, (doc_id, score) in enumerate(res.get(sid, []), start=1):
                    rows.append((sid, doc_id, score))
            if rows:
                _write_csv_triples(f_sc, rows, append=False); written.append(str(f_sc))

            # PROP targets (only if prop exists)
            if self.build_props and self._index_exists(enc, "prop"):
                # QUERY -> PROP
                res = self._search_bm25("prop", queries_q, tk) if enc.lower()=="bm25" else self._search_vec(enc,"prop",queries_q,tk)
                rows = []
                for qid, _ in queries_q:
                    for rank, (doc_id, score) in enumerate(res.get(qid, []), start=1):
                        rows.append((qid, doc_id, score))
                if rows:
                    _write_csv_triples(f_qp, rows, append=False); written.append(str(f_qp))

                # SUBQUERY -> PROP
                res = self._search_bm25("prop", queries_subq, tk) if enc.lower()=="bm25" else self._search_vec(enc,"prop",queries_subq,tk)
                rows = []
                for sid, _ in queries_subq:
                    for rank, (doc_id, score) in enumerate(res.get(sid, []), start=1):
                        rows.append((sid, doc_id, score))
                if rows:
                    _write_csv_triples(f_sp, rows, append=False); written.append(str(f_sp))

        return {"files": written}

    # -------------------------
    # Corpus
    # -------------------------
    def _load_full_corpus(self) -> List[dict]:
        if self._full_corpus is not None:
            return self._full_corpus

        p = self.corpus_path
        docs: List[dict] = []
        if p.is_file() and p.suffix.lower() == ".jsonl":
            docs = _read_jsonl(p)
        elif p.is_dir():
            corpus_dir = p / "corpus"
            if corpus_dir.exists():
                files = sorted(corpus_dir.glob("*.jsonl"))
                if not files:
                    raise FileNotFoundError(f"No files in {corpus_dir}")
                for f in files:
                    docs.extend(_read_jsonl(f))
            else:
                files = sorted(p.glob("*.jsonl"))
                if not files:
                    raise FileNotFoundError(f"No .jsonl files under {p}")
                for f in files:
                    docs.extend(_read_jsonl(f))
        else:
            raise FileNotFoundError(f"Corpus path not found or unsupported: {p}")

        norm = []
        for i, d in enumerate(docs):
            contents = (d.get("contents") or "").strip()
            if not contents:
                continue
            did = str(d.get("id") or f"doc_{i}")
            norm.append({"id": did, "text": contents})
        if not norm:
            raise ValueError("Empty corpus after loading.")
        self._full_corpus = norm

        cache = self.dataset_dir / "corpus_full.jsonl"
        if not cache.exists():
            _write_jsonl(cache, norm)
        return norm

    def _load_or_build_props_corpus(self) -> List[dict]:
        if not self.build_props:
            return []
        if self._props_corpus is not None:
            return self._props_corpus

        props_path = self.dataset_dir / "corpus_props.jsonl"
        if props_path.exists():
            self.logger.info("Loading cached proposition corpus…")
            self._props_corpus = _read_jsonl(props_path)
            return self._props_corpus

        # propositionize corpus with T5
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.logger.info("Building proposition corpus… (first run is heavier)")
        tokenizer = AutoTokenizer.from_pretrained(PROP_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(PROP_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        def _prop_batch(pars: List[str]) -> List[str]:
            inputs = tokenizer(
                pars, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            return tokenizer.batch_decode(out, skip_special_tokens=True)

        full = self._load_full_corpus()
        props: List[dict] = []
        for d in tqdm(full, desc="Propositionizing corpus"):
            # simple paragraph split
            paragraphs = [p.strip() for p in d["text"].split("\n\n") if len(p.split()) >= 10]
            pid = 0
            for i in range(0, len(paragraphs), self.batch_size):
                outs = _prop_batch(paragraphs[i:i+self.batch_size])
                for t in outs:
                    t = t.strip()
                    if not t:
                        continue
                    # append -<prop_idx> to existing id (works even if id already has -<chunk_idx>)
                    props.append({"id": f"{d['id']}-{pid}", "text": t, "source_id": d["id"]})
                    pid += 1

        if not props:
            self.logger.warning("No propositions produced; skipping props corpus.")
            props = []

        _write_jsonl(props_path, props)
        self._props_corpus = props
        return props

    # -------------------------
    # Queries
    # -------------------------
    def _load_queries(self, queries: Any) -> List[dict]:
        """
        Accepts:
          - path to .jsonl with rows {"id": "...", "title": "..."}
          - list[dict] with same schema
        Returns list of {"id", "title"}.
        """
        if isinstance(queries, (str, Path)):
            qpath = Path(queries)
            raw = _read_jsonl(qpath)
        elif isinstance(queries, list):
            raw = queries
        else:
            raise ValueError("queries must be a path or a list of dicts.")

        out = []
        for i, q in enumerate(raw):
            qid = str(q.get("id") or f"Q{i}")
            title = (q.get("title") or "").strip()
            if not title:
                continue
            out.append({"id": qid, "title": title})
        if not out:
            raise ValueError("No queries loaded.")
        return out

    def _prop_queries_same_model(self, queries: List[dict]) -> List[dict]:
        """
        Propositionize queries with the same T5 model.
        Returns subqueries as [{"id": "orig#i", "title": "...", "parent_id": "orig"}].
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(PROP_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(PROP_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        def _gen(texts: List[str]) -> List[str]:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128)
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            return dec

        subs: List[dict] = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc="Propositionizing queries"):
            batch = queries[i:i+self.batch_size]
            outs = _gen([q["title"] for q in batch])
            for q, out_text in zip(batch, outs):
                # split the single generated text into usable subqueries
                # split on newlines / bullets / semicolons; keep >= 3 words
                cand = re.split(r"[\n;•\-]+", out_text)
                cleaned = [c.strip() for c in cand if len(c.strip().split()) >= 3]
                if not cleaned:
                    # fall back to the original title as a single subquery
                    cleaned = [q["title"]]
                for j, t in enumerate(cleaned):
                    subs.append({"id": f"{q['id']}#{j}", "title": t, "parent_id": q["id"]})
        return subs

    # -------------------------
    # Index I/O
    # -------------------------
    def _enc_dir(self, encoder: str, variant: str) -> Path:
        d = self.dataset_dir / f"{_sanitize(encoder)}__{variant}__{_sanitize(self.index_type)}"
        _ensure_dir(d)
        return d

    def _bm25_paths(self, encoder: str, variant: str) -> Tuple[Path, Path]:
        base = self._enc_dir(encoder, variant)
        return (base / "bm25.pkl", base / "bm25_ids.pkl")

    def _faiss_paths(self, encoder: str, variant: str) -> Tuple[Path, Path]:
        base = self._enc_dir(encoder, variant)
        return (base / "faiss.index", base / "doc_ids.pkl")

    def _index_exists(self, encoder: str, variant: str) -> bool:
        if encoder.lower() == "bm25":
            a, b = self._bm25_paths(encoder, variant)
            return a.exists() and b.exists()
        a, b = self._faiss_paths(encoder, variant)
        return a.exists() and b.exists()

    # -------------------------
    # Build indexes
    # -------------------------
    def _build_bm25(self, docs: List[dict], encoder: str, variant: str) -> None:
        self.logger.info(f"Building BM25 for {variant}…")
        tokens, ids = [], []
        for d in docs:
            txt = d["text"]
            if not txt:
                continue
            tokens.append(txt.lower().split())
            ids.append(d["id"])
        if not tokens:
            raise ValueError("No texts for BM25.")
        bm25 = BM25Okapi(tokens)
        pkl, idp = self._bm25_paths(encoder, variant)
        with pkl.open("wb") as f:
            pickle.dump(bm25, f)
        with idp.open("wb") as f:
            pickle.dump(ids, f)

    def _build_vec_index(self, docs: List[dict], encoder: str, variant: str) -> None:
        self.logger.info(f"Building vector index for {encoder} ({variant})…")
        model = SentenceTransformer(encoder)

        texts, ids = [], []
        for d in docs:
            t = d["text"]
            if not t:
                continue
            texts.append(t)
            ids.append(d["id"])
        if not texts:
            raise ValueError("No texts for embeddings.")

        parts = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch = texts[i:i+self.batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            parts.append(emb.astype("float32", copy=False))
        embs = np.vstack(parts)
        _norm_l2(embs)  # cosine via IP

        d = embs.shape[1]
        n = embs.shape[0]

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(d)
        elif self.index_type == "ivf":
            nlist = _ivf_nlist(n)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        if hasattr(index, "is_trained") and not index.is_trained:
            self.logger.info("Training index…")
            index.train(embs)

        self.logger.info("Adding vectors…")
        index.add(embs)

        ipath, idpath = self._faiss_paths(encoder, variant)
        faiss.write_index(index, str(ipath))
        with idpath.open("wb") as f:
            pickle.dump(ids, f)

    def _ensure_index_one(self, encoder: str, variant: str, docs: List[dict]) -> None:
        if self._index_exists(encoder, variant):
            return
        if encoder.lower() == "bm25":
            self._build_bm25(docs, encoder, variant)
        else:
            self._build_vec_index(docs, encoder, variant)

    def ensure_all_indexes(self) -> None:
        full = self._load_full_corpus()
        props = self._load_or_build_props_corpus() if self.build_props else []
        for enc in self.encoders:
            self._ensure_index_one(enc, "chunk", full)
            if self.build_props and props:
                self._ensure_index_one(enc, "prop", props)

    # -------------------------
    # Search helpers
    # -------------------------
    def _search_bm25(
        self, variant: str, queries: List[Tuple[str, str]], top_k: int
    ) -> Dict[str, List[Tuple[str, float]]]:
        # queries: list of (query_id, text)
        pkl, idp = self._bm25_paths("bm25", variant)
        with pkl.open("rb") as f:
            bm25 = pickle.load(f)
        with idp.open("rb") as f:
            ids = pickle.load(f)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, qtext in queries:
            toks = qtext.lower().split()
            scores = bm25.get_scores(toks)
            k = min(top_k, len(scores))
            idxs = np.argpartition(scores, -k)[-k:]
            idxs = idxs[np.argsort(scores[idxs])[::-1]]
            out = [(ids[i], float(scores[i])) for i in idxs]
            results[qid] = out
        return results

    def _search_vec(
        self, encoder: str, variant: str, queries: List[Tuple[str, str]], top_k: int
    ) -> Dict[str, List[Tuple[str, float]]]:
        ipath, idp = self._faiss_paths(encoder, variant)
        index = faiss.read_index(str(ipath))
        with idp.open("rb") as f:
            ids = pickle.load(f)
        model = SentenceTransformer(encoder)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i+self.batch_size]
            texts = [qtext for _, qtext in batch]
            qemb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False).astype("float32")
            _norm_l2(qemb)
            D, I = index.search(qemb, min(top_k, len(ids)))
            for bi, (qid, _) in enumerate(batch):
                out = []
                for j, idx in enumerate(I[bi]):
                    if idx < 0 or idx >= len(ids):
                        continue
                    out.append((ids[idx], float(D[bi][j])))
                results[qid] = out
        return results

    def _doc_lookup(self, variant: str) -> Dict[str, str]:
        if variant == "chunk":
            corpus = self._load_full_corpus()
        else:
            corpus = self._load_or_build_props_corpus()
        return {d["id"]: d["text"] for d in corpus}

    # -------------------------
    # Public: 4-way search + dump
    # -------------------------
    def search_and_dump_4way(
        self,
        queries: Any,                       # path or list[{"id","title"}]
        max_subqueries: Optional[int] = None,  # unused now (propositionizer decides)
        save_snippets: bool = True,
        top_k: Optional[int] = None,
        run_tag: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Produces 4 files under indexes/<dataset>/runs/:

          1) subquery-prop   : subqueries (via T5) vs prop corpus
          2) query-chunk     : original queries vs chunk corpus
          3) subquery-chunk  : subqueries vs chunk corpus
          4) query-prop      : original queries vs prop corpus
        """
        self.ensure_all_indexes()
        q_list = self._load_queries(queries)
        subqs = self._prop_queries_same_model(q_list)

        doc_text_chunk = self._doc_lookup("chunk")
        doc_text_prop = self._doc_lookup("prop") if self.build_props else {}

        run_dir = self.dataset_dir / "runs"
        _ensure_dir(run_dir)
        tag = run_tag or _now_tag()

        files = {
            "subquery-prop": run_dir / f"subquery_prop_{tag}.jsonl",
            "query-chunk": run_dir / f"query_chunk_{tag}.jsonl",
            "subquery-chunk": run_dir / f"subquery_chunk_{tag}.jsonl",
            "query-prop": run_dir / f"query_prop_{tag}.jsonl",
        }
        tk = top_k or self.top_k

        # Pack query tuples for search
        queries_q = [(q["id"], q["title"]) for q in q_list]
        queries_subq = [(s["id"], s["title"]) for s in subqs]
        parent_map = {s["id"]: s["parent_id"] for s in subqs}
        q_text_map = {q["id"]: q["title"] for q in q_list}

        # Do searches per encoder and append to the right file
        for enc in self.encoders:
            # chunk variant targets
            # -- query-chunk
            self._search_and_append(
                enc, "chunk", queries_q, tk, files["query-chunk"],
                save_snippets, doc_text_chunk,
                extra_fields=lambda qid: {"query_id": qid, "query_text": q_text_map[qid]}
            )
            # -- subquery-chunk
            self._search_and_append(
                enc, "chunk", queries_subq, tk, files["subquery-chunk"],
                save_snippets, doc_text_chunk,
                extra_fields=lambda sid: {
                    "subquery_id": sid,
                    "subquery_text": [t for i,t in queries_subq if i == sid][0],
                    "parent_query_id": parent_map[sid],
                    "parent_query_text": q_text_map[parent_map[sid]],
                }
            )

            # prop variant targets (only if we have prop indexes)
            if self.build_props and doc_text_prop:
                # -- query-prop
                self._search_and_append(
                    enc, "prop", queries_q, tk, files["query-prop"],
                    save_snippets, doc_text_prop,
                    extra_fields=lambda qid: {"query_id": qid, "query_text": q_text_map[qid]}
                )
                # -- subquery-prop
                self._search_and_append(
                    enc, "prop", queries_subq, tk, files["subquery-prop"],
                    save_snippets, doc_text_prop,
                    extra_fields=lambda sid: {
                        "subquery_id": sid,
                        "subquery_text": [t for i,t in queries_subq if i == sid][0],
                        "parent_query_id": parent_map[sid],
                        "parent_query_text": q_text_map[parent_map[sid]],
                    }
                )

        return {k: str(v) for k, v in files.items()}

    def _search_and_append(
        self,
        encoder: str,
        variant: str,  # "chunk" | "prop"
        queries: List[Tuple[str, str]],
        top_k: int,
        outfile: Path,
        save_snippets: bool,
        doc_text_map: Dict[str, str],
        extra_fields,
    ):
        if not queries:
            return
        if encoder.lower() == "bm25":
            res = self._search_bm25(variant, queries, top_k)
        else:
            res = self._search_vec(encoder, variant, queries, top_k)

        rows = []
        for qid, _ in queries:
            hits = res.get(qid, [])
            for rank, (doc_id, score) in enumerate(hits, start=1):
                row = {
                    "encoder": encoder,
                    "target_variant": variant,      # "chunk" or "prop"
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": score,
                }
                if save_snippets:
                    row["snippet"] = _short_snippet(doc_text_map.get(doc_id, ""))
                # attach query/subquery fields
                row.update(extra_fields(qid))
                rows.append(row)
        if rows:
            _write_jsonl(outfile, rows, append=True)

# -----------------------------
# Simple CLI (optional)
# -----------------------------
def _parse_args():
    ap = argparse.ArgumentParser(description="Auto-index and 4-way retrieval.")
    ap.add_argument("--corpus", required=True, help="Path to corpus dir or .jsonl (id, contents)")
    ap.add_argument("--queries", required=True, help="Path to queries .jsonl (id, title)")
    ap.add_argument("--output", default="indexes", help="Output dir for indexes and runs")
    ap.add_argument("--encoders", default="default", help='Comma-separated list or "default"')
    ap.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-props", action="store_true", help="Skip proposition corpus/indexes")
    ap.add_argument("--snippets", action="store_true", help="Include snippets in output")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    encs = "default" if args.encoders == "default" else [e.strip() for e in args.encoders.split(",") if e.strip()]
    engine = MultiIndexRetriever(
        corpus_path=args.corpus,
        output_dir=args.output,
        encoders=encs,
        index_type=args.index_type,
        batch_size=args.batch_size,
        top_k=args.top_k,
        build_props=not args.no_props,
    )
    engine.ensure_all_indexes()
    files = engine.search_and_dump_4way(queries=args.queries, save_snippets=args.snippets)
    print(json.dumps(files, indent=2))
