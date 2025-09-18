#!/usr/bin/env python3
"""
generate_thrust_weights.py

Learn per-retriever (encoder × granularity) weights per query using a thrust-like
knowledge-bottleneck score, then merge all retrievers' results.

Compatible with outputs from multi_retriever.py (four run files). No edits required.
Optionally accepts encoder-overrides to let query/doc encoders differ.

USAGE
=====
python generate_thrust_weights.py \
  --runs paths=subquery_prop_20250917.jsonl,query_chunk_20250917.jsonl,subquery_chunk_20250917.jsonl,query_prop_20250917.jsonl \
  --dataset_dir indexes/<your_dataset_name> \
  --out_dir indexes/<your_dataset_name>/merges \
  --top_k 50 --softmax_temp 1.0 --max_workers 1

OUTPUTS
=======
- weights_<tag>.json          # per query per retriever weights
- merged_<tag>.jsonl          # merged ranking per query (doc_id, final_score, contributions)
"""

import os
import re
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Iterable
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

import faiss
from sentence_transformers import SentenceTransformer

# -----------------------
# Logging
# -----------------------
def _logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("thrust_weights")

# -----------------------
# IO helpers
# -----------------------
def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_csv_triples(path: Path) -> List[dict]:
    """
    CSV rows: qid,doc_id,score (no header).
    We infer retriever key from filename:
      <encsan>__<query|subquery>_<chunk|prop>__<tag>.csv
    retriever_key = encsan|<chunk|prop>|<query|subq>
    """
    fname = path.name
    m = re.match(r"(.+)__(query|subquery)_(chunk|prop)__(.+)\.csv$", fname)
    if not m:
        raise ValueError(f"Run filename not recognized: {fname}")
    encsan, qtype, variant, _ = m.groups()
    qtype = "subq" if qtype == "subquery" else "query"
    rkey = f"{encsan}|{variant}|{qtype}"

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            qid, did, score = ln.split(",", 2)
            rows.append({
                "retriever_key": rkey,
                "qid": qid,
                "doc_id": did,
                "score": float(score),
            })
    return rows


def _now_tag() -> str:
    import time
    return time.strftime("%Y%m%d_%H%M%S")

# -----------------------
# Embedding wrapper (cache models)
# -----------------------
class EmbedderCache:
    def __init__(self, default_model: str = "all-mpnet-base-v2"):
        self.models: Dict[str, SentenceTransformer] = {}
        self.default_model = default_model

    def get(self, name: Optional[str]) -> SentenceTransformer:
        key = name or self.default_model
        if key not in self.models:
            self.models[key] = SentenceTransformer(key)
        return self.models[key]

    def encode(self, model_name: Optional[str], texts: List[str], batch_size: int = 64) -> np.ndarray:
        m = self.get(model_name)
        out_parts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = m.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
            out_parts.append(emb.astype("float32", copy=False))
        embs = np.vstack(out_parts) if out_parts else np.zeros((0, 768), dtype="float32")
        faiss.normalize_L2(embs)
        return embs

# -----------------------
# Core
# -----------------------
def build_retriever_key(row: dict) -> str:
    """encoder|variant|query_type, where variant in {'chunk','prop'} and query_type {'query','subq'}."""
    encoder = row["encoder"]
    variant = row.get("target_variant") or row.get("variant") or "chunk"
    qtype = "subq" if "subquery_id" in row else "query"
    return f"{encoder}|{variant}|{qtype}"

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if math.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    if x.size == 0:
        return x
    x = x / max(1e-8, temp)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def infer_dataset_dir_from_runs(run_paths: List[Path]) -> Path:
    # runs are under: indexes/<dataset>/runs/<file>.jsonl
    # return indexes/<dataset>
    parents = set(p.parent.parent for p in run_paths)  # parent of "runs"
    if len(parents) == 1:
        return list(parents)[0]
    # fallback to the most common
    from collections import Counter
    cnt = Counter(parents)
    return cnt.most_common(1)[0][0]

def load_corpora(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    """Return {'chunk': {doc_id: text}, 'prop': {doc_id: text}}"""
    corpora: Dict[str, Dict[str, str]] = {"chunk": {}, "prop": {}}
    # chunk
    cf = dataset_dir / "corpus_full.jsonl"
    if cf.exists():
        for d in _read_jsonl(cf):
            did = str(d["id"])
            text = d.get("text") or d.get("contents") or ""
            corpora["chunk"][did] = text
    # prop
    pf = dataset_dir / "corpus_props.jsonl"
    if pf.exists():
        for d in _read_jsonl(pf):
            did = str(d["id"])
            text = d.get("text") or d.get("contents") or ""
            corpora["prop"][did] = text
    return corpora

def collect_runs(run_paths: List[Path]) -> List[dict]:
    rows = []
    for p in run_paths:
        if p.suffix.lower() == ".csv":
            rows.extend(read_csv_triples(p))
        elif p.suffix.lower() in [".trec", ".run"]:
            rows.extend(read_trec_run(p))  # if you kept TREC support
        else:
            rows.extend(_read_jsonl(p))   # legacy JSONL support
    return rows

def group_by_retriever(rows: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        buckets[build_retriever_key(r)].append(r)
    return buckets

def query_maps(rows: List[dict]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
      q_text[qid] = original query text
      subqs_by_parent[qid] = list of subquery texts for that parent
    """
    q_text: Dict[str, str] = {}
    subqs_by_parent: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if "query_id" in r:
            q_text[r["query_id"]] = r.get("query_text", q_text.get(r["query_id"], ""))
        if "parent_query_id" in r:
            pid = r["parent_query_id"]
            subqs_by_parent[pid].append(r.get("subquery_text", ""))
            # also keep parent query text if present
            if "parent_query_text" in r:
                q_text[pid] = r["parent_query_text"]
    # also catch cases from query-prop / query-chunk files
    for r in rows:
        if "query_id" in r and "query_text" in r:
            q_text[r["query_id"]] = r["query_text"]
    return q_text, subqs_by_parent

def kmeans_clusters(embs: np.ndarray, rng: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (centroids, sizes); number of clusters ~ n_docs**0.25 (>=3).
    """
    n = embs.shape[0]
    if n < 3:
        # degenerate
        return embs.copy(), np.ones((n,), dtype=np.int32)
    k = max(3, int(n ** 0.25))
    km = KMeans(n_clusters=min(k, n), random_state=rng, n_init="auto")
    labels = km.fit_predict(embs)
    centroids = km.cluster_centers_
    # sizes
    sizes = np.bincount(labels, minlength=centroids.shape[0])
    return centroids, sizes

def thrust_score(query_vec: np.ndarray, centroids: np.ndarray, sizes: np.ndarray, eps: float = 1e-8) -> float:
    """
    Thrust-like magnitude:
      For each centroid c with size s:
        w = s / (||q - c||^3 + eps)
        vector = w * (q - c)
      score = || sum(vector) ||
    """
    if centroids.size == 0:
        return 0.0
    diffs = (query_vec[None, :] - centroids)  # [k, d]
    dists = np.linalg.norm(diffs, axis=1) + eps
    weights = sizes / np.power(dists, 3)
    vec = (weights[:, None] * diffs).sum(axis=0)
    return float(np.linalg.norm(vec))

def normalize_scores_per_query(scores_for_query: Dict[str, float], method: str = "softmax", temp: float = 1.0) -> Dict[str, float]:
    keys = list(scores_for_query.keys())
    vals = np.array([scores_for_query[k] for k in keys], dtype="float32")
    if method == "minmax":
        n = minmax_norm(vals)
    else:
        n = softmax(vals, temp=temp)
    return {k: float(v) for k, v in zip(keys, n)}

def per_retriever_doc_embs(
    retr_rows: List[dict],
    corpora: Dict[str, Dict[str, str]],
    variant: str,
    embedder: EmbedderCache,
    ctx_model: Optional[str],
    batch_size: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Embed the set of unique doc_ids that this retriever returned (across all queries).
    """
    doc_ids: List[str] = []
    seen: Set[str] = set()
    for r in retr_rows:
        did = r["doc_id"]
        if did not in seen:
            seen.add(did); doc_ids.append(did)
    texts = [corpora[variant].get(did, "") for did in doc_ids]
    # drop empties
    keep = [(i, t) for i, t in enumerate(texts) if t]
    if not keep:
        return np.zeros((0, 768), dtype="float32"), []
    keep_idx, keep_txt = zip(*keep)
    embs = embedder.encode(ctx_model, list(keep_txt), batch_size=batch_size)
    doc_ids = [doc_ids[i] for i in keep_idx]
    return embs, doc_ids

def query_vectors_for_retriever(
    qids: List[str],
    q_text_map: Dict[str, str],
    subqs_by_parent: Dict[str, List[str]],
    embedder: EmbedderCache,
    qtype: str,
    query_model: Optional[str],
    batch_size: int
) -> np.ndarray:
    """
    For 'query' qtype: embed original query text.
    For 'subq'  qtype: embed all subqueries and average per parent (simple pooling).
    """
    if qtype == "query":
        texts = [q_text_map.get(qid, "") for qid in qids]
        return embedder.encode(query_model, texts, batch_size=batch_size)

    # subq: average subquery embeddings
    vecs = []
    for qid in qids:
        subtexts = [t for t in subqs_by_parent.get(qid, []) if t]
        if not subtexts:
            # fallback to original
            subtexts = [q_text_map.get(qid, "")]
        embs = embedder.encode(query_model, subtexts, batch_size=batch_size)
        if embs.shape[0] == 0:
            vecs.append(np.zeros((embs.shape[1] if embs.ndim == 2 else 768,), dtype="float32"))
        else:
            vecs.append(embs.mean(axis=0))
    return np.vstack(vecs)

def parse_encoder_overrides(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """
    Optional file (json/yaml) with per-encoder overrides:
    {
      "bm25": {"query_model": "all-mpnet-base-v2", "doc_model": "all-mpnet-base-v2"},
      "BAAI/bge-base-en-v1.5": {"query_model": "BAAI/bge-base-en-v1.5", "doc_model": "BAAI/bge-base-en-v1.5"}
    }
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in [".yaml", ".yml"]:
        import yaml
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_retriever_specs(
    grouped: Dict[str, List[dict]],
    overrides: Dict[str, Dict[str, str]],
    default_embedder: str
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    For each retriever_key: choose query/doc encoder models for embedding the KB.
    If no override is provided:
      - if encoder == 'bm25': use default_embedder for both query/doc
      - else: use the encoder name (same for both)
    Returns: spec[retriever_key] = {"encoder": enc, "variant": variant, "qtype": qtype, "query_model": "...", "doc_model": "..."}
    """
    specs: Dict[str, Dict[str, Optional[str]]] = {}
    for key in grouped.keys():
        enc, variant, qtype = key.split("|")
        ovr = overrides.get(enc, {})
        if enc.lower() == "bm25":
            q_model = ovr.get("query_model", default_embedder)
            d_model = ovr.get("doc_model", default_embedder)
        else:
            q_model = ovr.get("query_model", enc)
            d_model = ovr.get("doc_model", enc)
        specs[key] = {
            "encoder": enc, "variant": variant, "qtype": qtype,
            "query_model": q_model, "doc_model": d_model
        }
    return specs

def build_query_set(rows: List[dict]) -> List[str]:
    """
    We compute weights per ORIGINAL query id.
    """
    qids: Set[str] = set()
    for r in rows:
        if "parent_query_id" in r:
            qids.add(r["parent_query_id"])
        elif "query_id" in r:
            qids.add(r["query_id"])
    return sorted(qids)

def normalize_per_retriever_per_query_scores(hits: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Min-max normalize scores for a query within a retriever (doc-level)
    """
    if not hits:
        return {}
    scores = np.array([s for _, s in hits], dtype="float32")
    if len(scores) == 1:
        n = np.array([1.0], dtype="float32")
    else:
        n = minmax_norm(scores)
    return {doc: float(ns) for (doc, _), ns in zip(hits, n)}

def merge_rankings_for_query(
    per_retriever: Dict[str, Dict[str, float]],
    weights_for_query: Dict[str, float]
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    per_retriever: retriever_key -> {doc_id: score_norm}
    weights_for_query: retriever_key -> weight
    Returns ranked list of (doc_id, final_score, contributions_by_retriever)
    """
    docs: Set[str] = set()
    for dmap in per_retriever.values():
        docs.update(dmap.keys())
    out = []
    for did in docs:
        s = 0.0
        contrib = {}
        for rk, dmap in per_retriever.items():
            w = weights_for_query.get(rk, 0.0)
            v = dmap.get(did, 0.0)
            part = w * v
            contrib[rk] = part
            s += part
        out.append((did, s, contrib))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# -----------------------
# Main driver
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Per-retriever thrust weights + merged ranking.")
    ap.add_argument("--runs", required=True,
                    help="Comma-separated paths to run files from multi_retriever (any subset/all four).")
    ap.add_argument("--dataset_dir", required=False,
                    help="Path to indexes/<dataset_name>. If omitted, inferred from --runs.")
    ap.add_argument("--out_dir", required=False,
                    help="Where to write outputs. Defaults to <dataset_dir>/merges")
    ap.add_argument("--default_embedder", default="all-mpnet-base-v2",
                    help="Fallback encoder for BM25 or unspecified models.")
    ap.add_argument("--encoder_overrides", default=None,
                    help="JSON/YAML with per-encoder query_model/doc_model overrides.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=50, help="Top-K per retriever per query to read/normalize.")
    ap.add_argument("--score_norm", choices=["minmax"], default="minmax",
                    help="Per-retriever per-query score normalization (doc-level).")
    ap.add_argument("--weight_norm", choices=["softmax", "minmax"], default="softmax",
                    help="How to turn KB scores into weights per query.")
    ap.add_argument("--softmax_temp", type=float, default=1.0)
    ap.add_argument("--max_workers", type=int, default=None,
                    help="Parallel retriever jobs. Default=1 if CUDA visible, else #CPUs.")
    args = ap.parse_args()

    log = _logger()

    run_paths = [Path(p.strip()) for p in args.runs.split(",") if p.strip()]
    if not run_paths:
        log.error("No run files provided.")
        return

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = infer_dataset_dir_from_runs(run_paths)

    out_dir = Path(args.out_dir) if args.out_dir else (dataset_dir / "merges")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    # Load runs & corpora
    rows = collect_runs(run_paths)
    if not rows:
        log.error("Run files are empty.")
        return
    corpora = load_corpora(dataset_dir)
    q_text_map, subqs_by_parent = query_maps(rows)

    # Group by retriever and prep specs
    grouped = group_by_retriever(rows)
    overrides = parse_encoder_overrides(args.encoder_overrides)
    specs = build_retriever_specs(grouped, overrides, args.default_embedder)

    # Query universe (original ids)
    qids = build_query_set(rows)
    log.info(f"{len(qids)} original queries; {len(grouped)} retrievers found.")

    # Embed/cluster per retriever (optionally parallel)
    # (We keep it sequential by default to be GPU-friendly.)
    try:
        import torch
        cuda_present = torch.cuda.is_available()
    except Exception:
        cuda_present = False

    if args.max_workers is None:
        max_workers = 1 if cuda_present else os.cpu_count() or 1
    else:
        max_workers = max(1, args.max_workers)

    embed_cache = EmbedderCache(default_model=args.default_embedder)

    # Structures to hold:
    #   retriever_doc_norm_scores[qid][retriever_key][doc_id] = normalized doc score
    retriever_doc_norm_scores: Dict[str, Dict[str, Dict[str, float]]] = {qid: {} for qid in qids}
    #   kb_scores[qid][retriever_key] = thrust magnitude (before normalization to weights)
    kb_scores: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}

    # Worker loop (sequential by default)
    for rk, retr_rows in tqdm(grouped.items(), desc="Per-retriever processing"):
        enc, variant, qtype = rk.split("|")
        spec = specs[rk]
        doc_model = spec["doc_model"]
        query_model = spec["query_model"]

        # Collect per query top-K hits for doc-score normalization
        # hits_per_qid[qid] = [(doc_id, score), ...]
        hits_per_qid: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for r in retr_rows:
            qid = r.get("parent_query_id") or r.get("query_id")
            if not qid:
                continue
            hits_per_qid[qid].append((r["doc_id"], float(r["score"])))
        # Keep top_k per query
        for qid, hits in hits_per_qid.items():
            hits.sort(key=lambda x: x[1], reverse=True)
            hits_per_qid[qid] = hits[:args.top_k]

        # Normalize scores per query (doc-level)
        for qid, hits in hits_per_qid.items():
            retriever_doc_norm_scores[qid][rk] = normalize_per_retriever_per_query_scores(hits)

        # Build doc embedding corpus for this retriever (all unique docs returned)
        doc_embs, doc_ids = per_retriever_doc_embs(
            retr_rows, corpora, variant, embed_cache, doc_model, args.batch_size
        )
        if doc_embs.shape[0] == 0:
            # degenerate: all KB scores 0 for this retriever
            for qid in qids:
                kb_scores[qid][rk] = 0.0
            continue

        # Cluster to centroids
        cents, sizes = kmeans_clusters(doc_embs)

        # Build query vectors for this retriever
        qvecs = query_vectors_for_retriever(
            qids, q_text_map, subqs_by_parent, embed_cache, qtype, query_model, args.batch_size
        )

        # Compute thrust KB per query
        for i, qid in enumerate(qids):
            kb = thrust_score(qvecs[i], cents, sizes, eps=1e-8)
            kb_scores[qid][rk] = kb

    # Turn KB scores into weights per query
    weights: Dict[str, Dict[str, float]] = {}
    for qid in qids:
        weights[qid] = normalize_scores_per_query(
            kb_scores[qid],
            method=("minmax" if args.weight_norm == "minmax" else "softmax"),
            temp=args.softmax_temp
        )

    # Merge rankings per query using weights (missing docs -> 0)
    merged_rows = []
    for qid in qids:
        final_rank = merge_rankings_for_query(retriever_doc_norm_scores[qid], weights[qid])
        for rank, (doc_id, score, contrib) in enumerate(final_rank, start=1):
            merged_rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "final_score": score,
                "contrib": contrib
            })

    # Save
    weights_path = out_dir / f"weights_{tag}.json"
    merged_path = out_dir / f"merged_{tag}.jsonl"
    _write_json(weights_path, {
        "weights": weights,
        "kb_raw": kb_scores,
        "retrievers": list(grouped.keys()),
        "meta": {
            "default_embedder": args.default_embedder,
            "encoder_overrides": overrides,
            "top_k": args.top_k,
            "weight_norm": args.weight_norm,
            "softmax_temp": args.softmax_temp
        }
    })
    _write_jsonl(merged_path, merged_rows)

    print(json.dumps({
        "weights_path": str(weights_path),
        "merged_path": str(merged_path),
        "num_queries": len(qids),
        "num_retrievers": len(grouped),
        "notes": "Weights are per original query id. Retriever key = encoder|variant|query_type."
    }, indent=2))


if __name__ == "__main__":
    main()
