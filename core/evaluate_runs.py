#!/usr/bin/env python3
"""
evaluate_runs.py  (JSONL-only run; BEIR/TREC qrels)

Metrics: MAP, NDCG@K, P@K, Recall@K.

Usage:
  python core/evaluate_runs.py \
    --qrels /path/to/qrels.test.tsv \
    --run   /path/to/merged_YYYYmmdd.jsonl \
    --metrics ndcg@10,map,p@10,recall@100
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """
    Accepts:
      - BEIR-style TSV: qid \t docid \t rel
      - TREC qrels:     qid iter docid rel
    Returns: qrels[qid][docid] = relevance (int)
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i == 0:
                continue
            parts = ln.strip().split()
            if not parts: continue
            if len(parts) == 3:
                qid, doc, rel = parts[0], parts[1], int(parts[2])
            elif len(parts) >= 4:
                qid, doc, rel = parts[0], parts[2], int(parts[3])
            else:
                raise ValueError(f"Unrecognized qrels line: {ln}")
            qrels.setdefault(qid, {})[doc] = rel
    return qrels

def read_run_jsonl(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    runs: Dict[str, List[Tuple[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            obj = json.loads(ln)
            qid = obj["query_id"]; did = obj["doc_id"]; score = float(obj.get("final_score", 0.0))
            runs.setdefault(qid, []).append((did, score))
    for qid in runs:
        runs[qid].sort(key=lambda x: (-x[1], x[0]))
    return runs

# ---- metrics ----
def precision_at_k(ranked: List[str], relset: set, k: int) -> float:
    if k <= 0: return 0.0
    topk = ranked[:k]
    return sum(1 for d in topk if d in relset) / k

def recall_at_k(ranked: List[str], relset: set, k: int) -> float:
    if not relset: return 0.0
    topk = ranked[:k]
    return sum(1 for d in topk if d in relset) / len(relset)

def average_precision(ranked: List[str], relset: set) -> float:
    if not relset: return 0.0
    cum, hit = 0.0, 0
    for i, d in enumerate(ranked, start=1):
        if d in relset:
            hit += 1
            cum += hit / i
    return cum / len(relset)

def dcg_at_k(ranked: List[str], relmap: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        rel = relmap.get(doc, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranked: List[str], relmap: Dict[str, int], k: int) -> float:
    dcg = dcg_at_k(ranked, relmap, k)
    if not relmap: return 0.0
    ideal = sorted(relmap.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0

def main():
    ap = argparse.ArgumentParser(description="Evaluate JSONL run against qrels.")
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--metrics", default="ndcg@10,map,p@10,recall@100",
                    help="Comma-separated: ndcg@K,map,p@K,recall@K")
    ap.add_argument("--per_query", action="store_true")
    args = ap.parse_args()

    qrels = read_qrels(Path(args.qrels))
    runs = read_run_jsonl(Path(args.run))

    metric_specs = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    perq = {}

    for qid, relmap in qrels.items():
        ranked = [d for d, _ in runs.get(qid, [])]
        relset = {d for d, r in relmap.items() if r > 0}
        print(ranked, relset)
        perq[qid] = {}
        for spec in metric_specs:
            if spec == "map":
                perq[qid]["map"] = average_precision(ranked, relset)
            elif spec.startswith("p@"):
                k = int(spec.split("@",1)[1]); perq[qid][f"p@{k}"] = precision_at_k(ranked, relset, k)
            elif spec.startswith("recall@"):
                k = int(spec.split("@",1)[1]); perq[qid][f"recall@{k}"] = recall_at_k(ranked, relset, k)
            elif spec.startswith("ndcg@"):
                k = int(spec.split("@",1)[1]); perq[qid][f"ndcg@{k}"] = ndcg_at_k(ranked, relmap, k)
            else:
                raise ValueError(f"Unknown metric: {spec}")

    macro = {}
    for spec in metric_specs:
        key = "map" if spec == "map" else spec
        vals = [perq[qid][key] for qid in perq if key in perq[qid]]
        macro[key] = sum(vals) / len(vals) if vals else 0.0

    out = {"macro": macro}
    if args.per_query: out["per_query"] = perq
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
