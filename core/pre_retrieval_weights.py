
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import faiss
from sklearn.cluster import KMeans
from tqdm import tqdm

from core.utils import (
    _logger, _now_tag, _read_jsonl, _write_json,
    read_csv_triples, load_queries_whole, load_subqueries_multi
)
from core.encoders import EmbedderCache

def load_corpora(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    corpora: Dict[str, Dict[str, str]] = {"chunk": {}, "prop": {}}
    cf = dataset_dir / "corpus_full.jsonl"
    if cf.exists():
        for d in _read_jsonl(cf):
            did = str(d["id"]); txt = d.get("text") or d.get("contents") or ""
            corpora["chunk"][did] = txt
    pf = dataset_dir / "corpus_props.jsonl"
    if pf.exists():
        for d in _read_jsonl(pf):
            did = str(d["id"]); txt = d.get("text") or d.get("contents") or ""
            corpora["prop"][did] = txt
    return corpora


def collect_runs(run_paths: List[Path]) -> List[dict]:
    rows = []
    for p in run_paths:
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Expected CSV runs, got: {p}")
        rows.extend(read_csv_triples(p))  # expected to infer retriever_key from filename
    return rows


def build_retriever_key(row: dict) -> str:
    return row["retriever_key"]


def group_by_retriever(rows: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        buckets[build_retriever_key(r)].append(r)
    return buckets


def parse_encoder_overrides(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    return json.loads(p.read_text())


def build_retriever_specs(
    grouped: Dict[str, List[dict]],
    overrides: Dict[str, Dict[str, str]],
    default_embedder: str
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Map each retriever key to its (query_model, doc_model).
    Defaults: bm25 → default_embedder for both; others → encoder name (handled by EmbedderCache for DPR variants).
    retriever_key schema: "<encoder>|<chunk|prop>|<query|subq>"
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
        specs[key] = {"encoder": enc, "variant": variant, "qtype": qtype,
                      "query_model": q_model, "doc_model": d_model}
    return specs


def kmeans_clusters(embs: np.ndarray, rng: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    n = embs.shape[0]
    if n < 3:
        return embs.copy(), np.ones((n,), dtype=np.int32)
    k = max(3, int(n ** 0.25))
    km = KMeans(n_clusters=min(k, n), random_state=rng, n_init="auto")
    labels = km.fit_predict(embs)
    cents = km.cluster_centers_
    sizes = np.bincount(labels, minlength=cents.shape[0])
    return cents, sizes


def thrust_score(q: np.ndarray, cents: np.ndarray, sizes: np.ndarray, eps: float = 1e-8) -> float:
    if cents.size == 0:
        return 0.0
    diffs = (q[None, :] - cents)
    dists = np.linalg.norm(diffs, axis=1) + eps
    weights = sizes / np.power(dists, 3)
    vec = (weights[:, None] * diffs).sum(axis=0)
    return float(np.linalg.norm(vec))

def corpus_doc_embs_for_variant(
    corpora: Dict[str, Dict[str, str]],
    variant: str,
    embedder: EmbedderCache,
    ctx_model: Optional[str],
    batch_size: int,
    train_sample: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    texts = list(corpora.get(variant, {}).values())
    if not texts:
        return np.zeros((0, 768), dtype="float32")
    if train_sample is not None and len(texts) > train_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(texts), size=train_sample, replace=False)
        texts = [texts[i] for i in idx]
    return embedder.encode(ctx_model, texts, batch_size=batch_size, is_query=False)


def query_vectors_for_retriever(
    qids: List[str],
    q_text_map: Dict[str, str],
    subqs_by_parent: Dict[str, List[str]],
    embedder: EmbedderCache,
    qtype: str,
    query_model: Optional[str],
    batch_size: int
) -> np.ndarray:
    if qtype == "query":
        texts = [q_text_map.get(qid, "") for qid in qids]
        return embedder.encode(query_model, texts, batch_size=batch_size, is_query=True)
    vecs = []
    for qid in qids:
        subtexts = [t for t in subqs_by_parent.get(qid, []) if t] or [q_text_map.get(qid, "")]
        embs = embedder.encode(query_model, subtexts, batch_size=batch_size, is_query=True)
        if embs.shape[0] == 0:
            vecs.append(np.zeros((embs.shape[1] if embs.ndim == 2 else 768,), dtype="float32"))
        else:
            vecs.append(embs.mean(axis=0))
    return np.vstack(vecs)


def compute_pre_weights(
    runs: List[Path],
    queries: Path,
    dataset_dir: Path,
    out_dir: Path,
    default_embedder: str = "sentence-transformers/all-mpnet-base-v2",
    subqueries: Optional[Path] = None,
    encoder_overrides: Optional[Path] = None,
    batch_size: int = 64,
    train_sample: Optional[int] = None,
) -> Path:
    log = _logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    corpora = load_corpora(dataset_dir)
    q_text_map = load_queries_whole(queries)
    if subqueries:
        subqs_by_parent = load_subqueries_multi(subqueries)
    else:
        subqs_by_parent = {qid: [] for qid in q_text_map.keys()}

    rows = collect_runs(runs)
    grouped = group_by_retriever(rows)
    overrides = parse_encoder_overrides(str(encoder_overrides)) if encoder_overrides else {}
    specs = build_retriever_specs(grouped, overrides, default_embedder)

    qids = sorted(q_text_map.keys())
    log.info(f"{len(qids)} parent queries; {len(grouped)} retrievers detected.")

    embed_cache = EmbedderCache(default_model=default_embedder)
    kb_raw: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}

    for rk, _retr_rows in tqdm(grouped.items(), desc="KB per retriever"):
        enc, variant, qtype = rk.split("|")
        spec = specs[rk]
        ctx_model = spec["doc_model"]
        qry_model = spec["query_model"]

        train_embs = corpus_doc_embs_for_variant(
            corpora, variant, embed_cache, ctx_model, batch_size, train_sample=train_sample
        )
        if train_embs.shape[0] == 0:
            for qid in qids:
                kb_raw[qid][rk] = 0.0
            continue
        cents, sizes = kmeans_clusters(train_embs)
        qvecs = query_vectors_for_retriever(
            qids, q_text_map, subqs_by_parent, embed_cache, qtype, qry_model, batch_size
        )
        for i, qid in enumerate(qids):
            kb_raw[qid][rk] = thrust_score(qvecs[i], cents, sizes, eps=1e-8)

    weights_path = out_dir / f"weights_{tag}.json"
    _write_json(weights_path, {
        "type": "pre",
        "retrievers": list(grouped.keys()),
        "weights": kb_raw,
        "meta": {
            "note": "weights are RAW thrust/KB scores (no normalization).",
            "default_embedder": default_embedder,
            "encoder_overrides": overrides,
            "dataset_dir": str(dataset_dir),
            "runs": [str(p) for p in runs],
            "train_sample": train_sample,
        }
    })
    return weights_path



def main():
    ap = argparse.ArgumentParser(description="Compute RAW (unnormalized) pre-retrieval thrust/KB weights.")
    ap.add_argument("--runs", required=True, help="Comma-separated CSV run paths.")
    ap.add_argument("--queries", required=True, help="queries.whole.jsonl (id,title).")
    ap.add_argument("--subqueries", default=None, help="Optional: queries.multi.jsonl (id=parent#i,title).")
    ap.add_argument("--dataset_dir", required=True, help="indexes/<dataset> (expects corpus_full.jsonl / corpus_props.jsonl).")
    ap.add_argument("--out_dir", required=True, help="Where to write weights_*.json")
    ap.add_argument("--default_embedder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--encoder_overrides", default=None, help="JSON/YAML: encoder → {query_model, doc_model}")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--train_sample", type=int, default=None, help="Optional subsample size per variant for clustering.")
    args = ap.parse_args()

    log = _logger()
    run_paths = [Path(p.strip()) for p in args.runs.split(",") if p.strip()]
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    # Load corpora + queries (ALL parent queries; not limited by runs)
    corpora = load_corpora(dataset_dir)
    q_text_map = load_queries_whole(Path(args.queries))
    if args.subqueries:
        subqs_by_parent = load_subqueries_multi(Path(args.subqueries))
    else:
        subqs_by_parent = {qid: [] for qid in q_text_map.keys()}

    # Discover retrievers from CSV
    rows = collect_runs(run_paths)
    grouped = group_by_retriever(rows)
    overrides = parse_encoder_overrides(args.encoder_overrides)
    specs = build_retriever_specs(grouped, overrides, args.default_embedder)

    # Query universe: ALL parent ids from queries.whole.jsonl
    qids = sorted(q_text_map.keys())
    log.info(f"{len(qids)} parent queries; {len(grouped)} retrievers detected.")

    embed_cache = EmbedderCache(default_model=args.default_embedder)

    kb_raw: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}

    for rk, _retr_rows in tqdm(grouped.items(), desc="KB per retriever"):
        enc, variant, qtype = rk.split("|")
        spec = specs[rk]
        ctx_model = spec["doc_model"]
        qry_model = spec["query_model"]
        # Cluster on FULL corpus for the variant
        train_embs = corpus_doc_embs_for_variant(
            corpora, variant, embed_cache, ctx_model, args.batch_size,
            train_sample=args.train_sample
        )
        if train_embs.shape[0] == 0:
            for qid in qids:
                kb_raw[qid][rk] = 0.0
            continue
        cents, sizes = kmeans_clusters(train_embs)
        qvecs = query_vectors_for_retriever(
            qids, q_text_map, subqs_by_parent, embed_cache, qtype, qry_model, args.batch_size
        )
        for i, qid in enumerate(qids):
            kb_raw[qid][rk] = thrust_score(qvecs[i], cents, sizes, eps=1e-8)

    weights_path = out_dir / f"weights_{tag}.json"
    _write_json(weights_path, {
        "type": "pre",
        "retrievers": list(grouped.keys()),
        "weights": kb_raw,  # RAW, unnormalized
        "meta": {
            "note": "weights are RAW thrust/KB scores (no normalization).",
            "default_embedder": args.default_embedder,
            "encoder_overrides": overrides,
            "dataset_dir": str(dataset_dir),
            "runs": [str(p) for p in run_paths],
            "train_sample": args.train_sample,
        }
    })
    print(json.dumps({"weights_path": str(weights_path)}, indent=2))

if __name__ == "__main__":
    main()