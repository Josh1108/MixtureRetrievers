#!/usr/bin/env python3
# indexing.py — reconciled with DPR-aware encoders
from __future__ import annotations
import os, re, json, time, math, argparse, logging, pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import faiss
from rank_bm25 import BM25Okapi

# Shared encoders
from core.encoders import EmbedderCache, resolve_model_name

PROP_MODEL_NAME = "chentong00/propositionizer-wiki-flan-t5-large"

# -----------------------------
# Helpers
# -----------------------------

def _setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("indexing")

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
            if line:
                out.append(json.loads(line))
    return out

def _write_csv_triples(path: Path, rows: List[Tuple[str, str, float]], append: bool = False) -> None:
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
        encoders: Optional[List[str]] = None,
        index_type: str = "flat",
        dataset_name: Optional[str] = None,
        batch_size: int = 128,
        top_k: int = 50,
        build_props: bool = True,
        default_encoder: str = "sentence-transformers/all-mpnet-base-v2",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or _setup_logger()
        self.corpus_path = Path(corpus_path)
        self.output_root = Path(output_dir)
        self.index_type = index_type
        self.batch_size = int(batch_size)
        self.top_k = int(top_k)
        self.build_props = build_props
        self.embed = EmbedderCache(default_model=default_encoder)

        if dataset_name:
            self.dataset_name = _sanitize(dataset_name)
        elif self.corpus_path.is_dir():
            if self.corpus_path.name == "corpus" and self.corpus_path.parent.name:
                self.dataset_name = _sanitize(self.corpus_path.parent.name)
            else:
                self.dataset_name = _sanitize(self.corpus_path.name)
        else:
            self.dataset_name = _sanitize(self.corpus_path.stem)

        self.dataset_dir = self.output_root / self.dataset_name
        _ensure_dir(self.dataset_dir)

        # If encoders not provided, use a sensible default set
        self.encoders = list(encoders) if encoders else [
            # Sparse
            "bm25",
            # Popular dense encoders (aliases supported via encoders.py)
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "multi-qa-mpnet-base-dot-v1",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            # Friendly aliases
            "SimCSE",
            "Contriever",
            "DPR",
            "ANCE",
            "GTR-T5-base",
            "MPNet",
        ]

        self._full_corpus: Optional[List[dict]] = None
        self._props_corpus: Optional[List[dict]] = None

    def _corpus_dir(self) -> Path:
        return self.corpus_path if self.corpus_path.is_dir() else self.corpus_path.parent

    # -------------------------
    # Public APIs for runs
    # -------------------------
    def search_and_dump_4way_csv_ids(
        self,
        queries: Any,
        top_k: Optional[int] = None,
        run_tag_prefix: Optional[str] = None,
        subqueries: Optional[Any] = None,
    ) -> Dict[str, List[str]]:
        """
        Writes per-encoder CSV files using the filename convention expected by
        pre_retrieval_weights.py to infer retriever_key from the filename:
          <encoder>__<query|subquery>_<chunk|prop>__<tag>.csv
        Each CSV row: qid,doc_id,score (no header)
        """
        self.ensure_all_indexes()
        q_list = self._load_queries(queries)
        if subqueries is not None:
            subqs = self._load_queries(subqueries)
            for s in subqs:
                if "parent_id" not in s:
                    s["parent_id"] = s["id"].split("#", 1)[0]
        else:
            subqs = self._prop_queries_same_model(q_list)

        tk = top_k or self.top_k
        tag = run_tag_prefix or _now_tag()
        run_dir = self.dataset_dir / "runs"
        _ensure_dir(run_dir)

        queries_q = [(q["id"], q["title"]) for q in q_list]
        queries_subq = [(s["id"], s["title"]) for s in subqs]

        written: List[str] = []
        for enc in tqdm(self.encoders):
            enc_base = _sanitize(enc)
            f_qc = run_dir / f"{enc_base}__query_chunk__{tag}.csv"
            f_sc = run_dir / f"{enc_base}__subquery_chunk__{tag}.csv"
            f_qp = run_dir / f"{enc_base}__query_prop__{tag}.csv"
            f_sp = run_dir / f"{enc_base}__subquery_prop__{tag}.csv"
            print(f"--- Searching & writing for {enc} ---")
            # QUERY → CHUNK
            res = self._search("chunk", enc, queries_q, tk)
            rows = [(qid, did, score) for qid, _ in queries_q for (did, score) in res.get(qid, [])]
            if rows: _write_csv_triples(f_qc, rows); written.append(str(f_qc))
            print(f"  Wrote {len(rows)} rows to {f_qc}")
            # SUBQUERY → CHUNK
            res = self._search("chunk", enc, queries_subq, tk)
            rows = [(sid, did, score) for sid, _ in queries_subq for (did, score) in res.get(sid, [])]
            if rows: _write_csv_triples(f_sc, rows); written.append(str(f_sc))
            print(f"  Wrote {len(rows)} rows to {f_sc}")
            # PROP targets (if exist)
            if self.build_props and self._index_exists(enc, "prop"):
                res = self._search("prop", enc, queries_q, tk)
                rows = [(qid, did, score) for qid, _ in queries_q for (did, score) in res.get(qid, [])]
                if rows: _write_csv_triples(f_qp, rows); written.append(str(f_qp))

                res = self._search("prop", enc, queries_subq, tk)
                rows = [(sid, did, score) for sid, _ in queries_subq for (did, score) in res.get(sid, [])]
                if rows: _write_csv_triples(f_sp, rows); written.append(str(f_sp))

        return {"files": written}

    def search_and_dump_4way(
        self,
        queries: Any,
        save_snippets: bool = True,
        top_k: Optional[int] = None,
        run_tag: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        JSONL variant (kept from your original) — file names do NOT carry retriever_key
        but contents include encoder/variant, so useful for inspection.
        """
        self.ensure_all_indexes()
        q_list = self._load_queries(queries)
        subqs = self._prop_queries_same_model(q_list)

        doc_text_chunk = self._doc_lookup("chunk")
        doc_text_prop  = self._doc_lookup("prop") if self.build_props else {}

        run_dir = self.dataset_dir / "runs"; _ensure_dir(run_dir)
        tag = run_tag or _now_tag()
        files = {
            "subquery-prop":  run_dir / f"subquery_prop_{tag}.jsonl",
            "query-chunk":    run_dir / f"query_chunk_{tag}.jsonl",
            "subquery-chunk": run_dir / f"subquery_chunk_{tag}.jsonl",
            "query-prop":     run_dir / f"query_prop_{tag}.jsonl",
        }
        tk = top_k or self.top_k

        queries_q    = [(q["id"], q["title"]) for q in q_list]
        queries_subq = [(s["id"], s["title"]) for s in subqs]
        parent_map   = {s["id"]: (s.get("parent_id") or s["id"].split("#",1)[0]) for s in subqs}
        q_text_map   = {q["id"]: q["title"] for q in q_list}

        for enc in self.encoders:
            # chunk
            self._search_and_append(enc, "chunk", queries_q, tk, files["query-chunk"], save_snippets, doc_text_chunk,
                extra_fields=lambda qid: {"query_id": qid, "query_text": q_text_map[qid]})
            self._search_and_append(enc, "chunk", queries_subq, tk, files["subquery-chunk"], save_snippets, doc_text_chunk,
                extra_fields=lambda sid: {
                    "subquery_id": sid,
                    "subquery_text": [t for i,t in queries_subq if i == sid][0],
                    "parent_query_id": parent_map[sid],
                    "parent_query_text": q_text_map[parent_map[sid]],
                })
            # prop
            if self.build_props and doc_text_prop:
                self._search_and_append(enc, "prop", queries_q, tk, files["query-prop"], save_snippets, doc_text_prop,
                    extra_fields=lambda qid: {"query_id": qid, "query_text": q_text_map[qid]})
                self._search_and_append(enc, "prop", queries_subq, tk, files["subquery-prop"], save_snippets, doc_text_prop,
                    extra_fields=lambda sid: {
                        "subquery_id": sid,
                        "subquery_text": [t for i,t in queries_subq if i == sid][0],
                        "parent_query_id": parent_map[sid],
                        "parent_query_text": q_text_map[parent_map[sid]],
                    })
        return {k: str(v) for k, v in files.items()}

    # -------------------------
    # Internals: corpus & props
    # -------------------------
    def _load_full_corpus(self) -> List[dict]:
        if self._full_corpus is not None:
            return self._full_corpus
        corpus_dir = self._corpus_dir()
        chunk_path = corpus_dir / "corpus.chunk.jsonl"
        docs: List[dict] = []
        if chunk_path.exists():
            raws = _read_jsonl(chunk_path)
        else:
            p = self.corpus_path
            if p.is_file() and p.suffix.lower() == ".jsonl":
                raws = _read_jsonl(p)
            elif p.is_dir():
                corpus_dir2 = p / "corpus"
                files = sorted((corpus_dir2 if corpus_dir2.exists() else p).glob("*.jsonl"))
                if not files:
                    raise FileNotFoundError(f"No .jsonl files in {p}")
                raws = []
                for f in files:
                    raws.extend(_read_jsonl(f))
            else:
                raise FileNotFoundError(f"Corpus path not found or unsupported: {p}")
        for i, d in enumerate(raws):
            text = (d.get("contents") or d.get("text") or "").strip()
            if not text:
                continue
            did = str(d.get("id") or f"doc_{i}")
            docs.append({"id": did, "text": text})
        if not docs:
            raise ValueError("Empty corpus after loading.")
        self._full_corpus = docs
        cache = self.dataset_dir / "corpus_full.jsonl"
        if not cache.exists():
            _write_jsonl(cache, docs)
        return docs

    def _load_or_build_props_corpus(self) -> List[dict]:
        if not self.build_props:
            return []
        if self._props_corpus is not None:
            return self._props_corpus
        corpus_dir = self._corpus_dir()
        local_prop = corpus_dir / "corpus.prop.jsonl"
        cached_prop = self.dataset_dir / "corpus_props.jsonl"
        if local_prop.exists():
            props = _read_jsonl(local_prop)
            norm = []
            for i, d in enumerate(props):
                t = (d.get("contents") or d.get("text") or "").strip()
                if not t:
                    continue
                did = str(d.get("id") or f"prop_{i}")
                norm.append({"id": did, "text": t})
            self._props_corpus = norm
            _write_jsonl(cached_prop, norm)
            return norm
        if cached_prop.exists():
            self._props_corpus = _read_jsonl(cached_prop)
            return self._props_corpus
        # Build once using T5 model
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(PROP_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(PROP_MODEL_NAME)
        device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
        model.to(device).eval()

        def _prop_batch(pars: List[str]) -> List[str]:
            inputs = tokenizer(pars, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256)
            return tokenizer.batch_decode(out, skip_special_tokens=True)

        full = self._load_full_corpus()
        props: List[dict] = []
        pid = 0
        for i in tqdm(range(0, len(full), self.batch_size), desc="Propositionizing corpus"):
            batch_docs = full[i:i+self.batch_size]
            paragraphs: List[str] = []
            ids: List[str] = []
            for d in batch_docs:
                pars = [p.strip() for p in d["text"].split("\n\n") if len(p.split()) >= 10]
                paragraphs.extend(pars)
                ids.extend([d["id"]]*len(pars))
            outs = _prop_batch(paragraphs) if paragraphs else []
            for src_id, t in zip(ids, outs):
                t = t.strip()
                if t:
                    props.append({"id": f"{src_id}-{pid}", "text": t, "source_id": src_id})
                    pid += 1
        _write_jsonl(local_prop, props)
        _write_jsonl(cached_prop, props)
        self._props_corpus = props
        return props

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
        texts, ids = [], []
        for d in docs:
            t = d["text"]
            if not t:
                continue
            texts.append(t)
            ids.append(d["id"])
        if not texts:
            raise ValueError("No texts for embeddings.")
        # DOC embeddings (ctx for DPR; standard ST elsewhere)
        embs = self.embed.encode(encoder, texts, batch_size=self.batch_size, is_query=False)
        d = embs.shape[1]; n = embs.shape[0]
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
            print(f"--- Ensuring index for {enc} ---")
            self._ensure_index_one(enc, "chunk", full)
            if self.build_props and props:
                self._ensure_index_one(enc, "prop", props)

    # -------------------------
    # Search helpers
    # -------------------------
    def _search(self, variant: str, encoder: str, queries: List[Tuple[str, str]], top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        if encoder.lower() == "bm25":
            return self._search_bm25(variant, queries, top_k)
        return self._search_vec(encoder, variant, queries, top_k)

    def _search_bm25(self, variant: str, queries: List[Tuple[str, str]], top_k: int) -> Dict[str, List[Tuple[str, float]]]:
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

    def _search_vec(self, encoder: str, variant: str, queries: List[Tuple[str, str]], top_k: int) -> Dict[str, List[Tuple[str, float]]]:
        ipath, idp = self._faiss_paths(encoder, variant)
        index = faiss.read_index(str(ipath))
        with idp.open("rb") as f:
            ids = pickle.load(f)
        results: Dict[str, List[Tuple[str, float]]] = {}
        # Encode queries with proper head (DPR question vs ST as usual)
        texts = [qtext for _, qtext in queries]
        qemb = self.embed.encode(encoder, texts, batch_size=self.batch_size, is_query=True).astype("float32")
        _norm_l2(qemb)
        D, I = index.search(qemb, min(top_k, len(ids)))
        for bi, (qid, _) in enumerate(queries):
            out = []
            for j, idx in enumerate(I[bi]):
                if idx < 0 or idx >= len(ids):
                    continue
                out.append((ids[idx], float(D[bi][j])))
            results[qid] = out
        return results

    def _doc_lookup(self, variant: str) -> Dict[str, str]:
        corpus = self._load_full_corpus() if variant == "chunk" else self._load_or_build_props_corpus()
        return {d["id"]: d["text"] for d in corpus}

    def _load_queries(self, queries: Any) -> List[dict]:
        if isinstance(queries, (str, Path)):
            raw = _read_jsonl(Path(queries))
        elif isinstance(queries, list):
            raw = queries
        else:
            raise ValueError("queries must be a path or a list of dicts.")
        out = []
        for i, q in enumerate(raw):
            qid = str(q.get("id") or f"Q{i}")
            title = (q.get("title") or "").strip()
            if title:
                out.append({"id": qid, "title": title})
        if not out:
            raise ValueError("No queries loaded.")
        return out

    def _prop_queries_same_model(self, queries: List[dict]) -> List[dict]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained(PROP_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(PROP_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        def _gen(texts: List[str]) -> List[str]:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128)
            return tokenizer.batch_decode(out, skip_special_tokens=True)
        subs: List[dict] = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc="Propositionizing queries"):
            batch = queries[i:i+self.batch_size]
            outs = _gen([q["title"] for q in batch])
            for q, out_text in zip(batch, outs):
                cand = re.split(r"[\n;•\-]+", out_text)
                cleaned = [c.strip() for c in cand if len(c.strip().split()) >= 3]
                if not cleaned:
                    cleaned = [q["title"]]
                for j, t in enumerate(cleaned):
                    subs.append({"id": f"{q['id']}#{j}", "title": t, "parent_id": q["id"]})
        return subs

    def _search_and_append(self, encoder: str, variant: str, queries: List[Tuple[str, str]], top_k: int, outfile: Path, save_snippets: bool, doc_text_map: Dict[str, str], extra_fields):
        if not queries:
            return
        res = self._search(variant, encoder, queries, top_k)
        rows = []
        for qid, _ in queries:
            hits = res.get(qid, [])
            for rank, (doc_id, score) in enumerate(hits, start=1):
                row = {"encoder": encoder, "target_variant": variant, "rank": rank, "doc_id": doc_id, "score": score}
                if save_snippets:
                    row["snippet"] = _short_snippet(doc_text_map.get(doc_id, ""))
                row.update(extra_fields(qid))
                rows.append(row)
        if rows:
            _write_jsonl(outfile, rows, append=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-index and DPR-aware 4-way retrieval.")
    ap.add_argument("--corpus", required=True, help="Path to corpus dir or .jsonl (id, contents)")
    ap.add_argument("--queries", required=True, help="Path to queries .jsonl (id, title)")
    ap.add_argument("--output", default="indexes", help="Output dir for indexes and runs")
    ap.add_argument("--encoders", default=None, help='Comma-separated list; if omitted uses built-ins')
    ap.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-props", action="store_true", help="Skip proposition corpus/indexes")
    ap.add_argument("--snippets", action="store_true", help="Include snippets in JSONL output")
    ap.add_argument("--default-encoder", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    encs = None if args.encoders is None else [e.strip() for e in args.encoders.split(",") if e.strip()]
    engine = MultiIndexRetriever(
        corpus_path=args.corpus,
        output_dir=args.output,
        encoders=encs,
        index_type=args.index_type,
        batch_size=args.batch_size,
        top_k=args.top_k,
        build_props=not args.no_props,
        default_encoder=args.default_encoder,
    )
    engine.ensure_all_indexes()
    files = engine.search_and_dump_4way(queries=args.queries, save_snippets=args.snippets)
    print(json.dumps(files, indent=2))