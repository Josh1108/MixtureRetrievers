"""fusion_eval.py
==================

script for loading per‑encoder retrieval results, merging them via
several fusion strategies, and reporting **NDCG@5/20** with BEIR’s
`EvaluateRetrieval` utility.

Major features
--------------
* Works with *chunk* vs *prop* document IDs and whole vs sub‑queries.
* Supports **max/mean** internal score merges, then **RRF / weighted‑sum /
  normalized‑sum** cross‑encoder fusion.
* Weight files are read from the *MixGR* JSON format and aligned per‑query.

Usage example
-------------
```bash
python zero_shot_fusion.py \
  --result-dir /path/to/outputs \
  --dataset-name scifact \
  --qrels-path /path/to/scifact/qrels/test.tsv \
  --ret-merge-method normalized_sum \
  --mixgr-merge-method max \
  --weights-dir /path/to/kb_weights
```
"""
import os
import pandas as pd
import csv
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODERS: Sequence[str] = (
    "simcse",
    "ance",
    "contriever",
    "tasb",
    "all-mpnet-base-v2",
    "gtr",
    "dpr",
    "bm25",
)
TOP_K: Sequence[int] = (5, 20)
RRF_K: int = 60


def extract_base_pid(pid,dataset_name):
    
    parts: List[str] = pid.split("-")

    if dataset_name == "nfcorpus":
        if len(parts) > 3:
            return "-".join(parts[:-2]), "prop"
        return "-".join(parts[:-1]), "chunk"

    if len(parts) > 2:  # non‑nfcorpus: >2 segments ⇒ prop
        return "-".join(parts[:-2]), "prop"
    return "-".join(parts[:-1]), "chunk"

def get_metrics(
    qid_pids: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per‑query NDCG@5/20."""
    evaluator = EvaluateRetrieval()
    ndcg5: Dict[str, float] = {}
    ndcg20: Dict[str, float] = {}

    for qid, retrieved in qid_pids.items():
        if qid not in qrels:
            raise KeyError(f"Missing qrels for qid {qid}")
        res = evaluator.evaluate({qid: qrels[qid]}, {qid: retrieved}, TOP_K)[0]
        ndcg5[qid] = res["NDCG@5"]
        ndcg20[qid] = res["NDCG@20"]

    return ndcg5, ndcg20


def _aggregate_scores(
    encoder_qid_pid_score: Mapping[str, Mapping[str, Mapping[str, float]]],
    dataset_name: str,
    mode: str = "max",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate chunk/prop duplicates via *max* or *mean* within each encoder."""

    merged: Dict[str, Dict[str, Dict[str, Tuple[float, int]]]] = {}

    for encoder, qid_pid_scores in encoder_qid_pid_score.items():
        for qid, pid_scores in qid_pid_scores.items():
            is_sub = "#" in qid
            base_qid = qid.split("#", 1)[0]

            for pid, score in pid_scores.items():
                base_pid, pid_type = extract_base_pid(pid, dataset_name)
                if pid_type not in {"chunk", "prop"}:
                    continue

                case = f"{pid_type}_{'sub' if is_sub else 'whole'}"
                enc_case = f"{encoder}_{case}"
                merged.setdefault(enc_case, {}).setdefault(base_qid, {})
                prev_score, cnt = merged[enc_case][base_qid].get(base_pid, (0.0, 0))
                if mode == "max":
                    new_score = max(prev_score, score)
                    cnt = max(cnt, 1)
                elif mode == "mean":
                    new_score = prev_score + score
                    cnt += 1
                else:
                    raise ValueError(f"Unknown aggregation mode: {mode}")
                merged[enc_case][base_qid][base_pid] = (new_score, cnt)

    # Finalise means
    final: Dict[str, Dict[str, Dict[str, float]]] = {}
    for enc_case, qid_pid in merged.items():
        final[enc_case] = {
            q: {p: s / c for p, (s, c) in pid_scores.items()}
            for q, pid_scores in qid_pid.items()
        }
    return final

def load_all_result_files(
    result_dir: Path,
    dataset_name: str,
    merge_method: str = "max",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load every encoder directory under *result_dir* and merge duplicates."""
    enc_qid_pid: Dict[str, Dict[str, Dict[str, float]]] = {}

    for enc in ENCODERS:
        enc_dir = result_dir / enc
        if not enc_dir.is_dir():
            logging.warning("Encoder directory not found: %s", enc_dir)
            continue
        enc_qid_pid[enc] = {}
        for fp in enc_dir.iterdir():
            if fp.name in {"rrf", "log"} or fp.suffix == ".log":
                continue
            for qid, pid, score in _read_result_records(fp):
                enc_qid_pid[enc].setdefault(qid, {})[pid] = max(
                    score, enc_qid_pid[enc][qid].get(pid, float("-inf"))
                )

    return _aggregate_scores(enc_qid_pid, dataset_name, merge_method)

def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load BEIR/trec‑style qrels TSV into a nested dict."""
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_file.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for query_id, corpus_id, score in reader:
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels

def _merge_query_weights(kb: Mapping[str, Sequence], mode: str = "max") -> Tuple[List[str], List[float]]:
    """Collapse sub‑query IDs using *max* or *mean* aggregation."""
    scores: Dict[str, Tuple[float, int]] = {}
    for qid, w in zip(kb.get("ids", []), kb.get("kb_list", [])):
        base = qid.split("#", 1)[0]
        prev, cnt = scores.get(base, (0.0, 0))
        if mode == "max":
            scores[base] = (max(prev, w), cnt + 1)
        else:  # mean
            scores[base] = (prev + w, cnt + 1)
    ids = list(scores)
    vals = [s if mode == "max" else s / c for s, c in scores.values()]
    return ids, vals


_WEIGHT_PATTERNS: Sequence[Tuple[str, str]] = (
    ("full_documents", "chunk_whole"),
    ("sub_documents", "prop_sub"),
    ("full_documents-sub_queries", "chunk_sub"),
    ("sub_documents-full_queries", "prop_whole"),
)


def load_encoder_weights(
    dataset_name: str,
    dir_path: Path,
    merge_type: str = "max",
) -> Dict[str, Tuple[List[float], List[str]]]:
    """Return mapping encoder_case → (weights, qids)."""
    enc_to_weight: Dict[str, Tuple[List[float], List[str]]] = {}

    for enc in ENCODERS:
        for suffix, case in _WEIGHT_PATTERNS:
            fp = dir_path / f"{enc}_{dataset_name}_{suffix}.json"
            if not fp.exists():
                continue
            with fp.open() as f:
                kb = json.load(f)
            ids, vals = (
                (kb.get("ids", []), kb.get("kb_list", []))
                if "sub" not in suffix
                else _merge_query_weights(kb, merge_type)
            )
            enc_to_weight[f"{enc}_{case}"] = (vals, ids)

    return enc_to_weight

def rrf_fusion(
    encoder_qid_pid_score: Mapping[str, Mapping[str, Mapping[str, float]]],
    k: int = RRF_K,
) -> Dict[str, Dict[str, float]]:
    """Reciprocal‑rank fusion."""
    fused: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, qid_pid_scores in encoder_qid_pid_score.items():
        for qid, pid_scores in qid_pid_scores.items():
            for rank, (pid, _) in enumerate(
                sorted(pid_scores.items(), key=lambda x: x[1], reverse=True),
                start=1,
            ):
                fused[qid][pid] += 1.0 / (k + rank)
    return {
        qid: dict(sorted(p.items(), key=lambda x: x[1], reverse=True))
        for qid, p in fused.items()
    }


def weighted_sum_fusion(
    encoder_qid_pid_score: Mapping[str, Mapping[str, Mapping[str, float]]],
    encoder_weights: Mapping[str, Tuple[Sequence[float], Sequence[str]]],
) -> Dict[str, Dict[str, float]]:
    """Weighted, per‑query normalised score sum."""
    fused: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for enc_case, qid_pid_scores in encoder_qid_pid_score.items():
        try:
            weights, qids = encoder_weights[enc_case]
        except KeyError:  # encoder not in weight file → skip
            logging.warning("Missing weights for %s – skipped", enc_case)
            continue

        for weight, qid in zip(weights, qids):
            pid_scores = qid_pid_scores.get(qid)
            if not pid_scores:
                continue
            ssum = sum(pid_scores.values()) or 1.0
            for pid, score in pid_scores.items():
                fused[qid][pid] += (score / ssum) * weight

    return {
        qid: dict(sorted(p.items(), key=lambda x: x[1], reverse=True))
        for qid, p in fused.items()
    }




def normalized_sum_fusion(
    encoder_qid_pid_score: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Sum of unit‑normalised scores across encoders."""
    fused: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_pids: Dict[str, set[str]] = defaultdict(set)

    for qid_pid_scores in encoder_qid_pid_score.values():
        for qid, pid_scores in qid_pid_scores.items():
            all_pids[qid].update(pid_scores)

    for qid_pid_scores in encoder_qid_pid_score.values():
        for qid, pid_scores in qid_pid_scores.items():
            ssum = sum(pid_scores.values()) or 1.0
            for pid in all_pids[qid]:
                fused[qid][pid] += pid_scores.get(pid, 0.0) / ssum
    return {
        qid: dict(sorted(p.items(), key=lambda x: x[1], reverse=True))
        for qid, p in fused.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion evaluation for BEIR runs")
    parser.add_argument("--result-dir", type=Path, required=True, help="Root dir containing encoder sub‑dirs with run files")
    parser.add_argument("--dataset-name", type=str, required=True, help="BEIR dataset name (e.g. scifact)")
    parser.add_argument("--qrels-path", type=Path, required=True, help="Path to qrels TSV (test split)")
    parser.add_argument("--ret-merge-method", choices=["normalized_sum", "weighted_sum", "rrf"], default="normalized_sum", help="Inter‑encoder fusion method")
    parser.add_argument("--mixgr-merge-method", choices=["max", "mean"], default="max", help="Intra‑encoder duplicate merge method")
    parser.add_argument("--weights-dir", type=Path, help="Directory containing MixGR weight JSONs (needed for weighted_sum)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.info("Args: %s", args)

    # ---------------------------------------------------------------------
    # Qrels loading
    # ---------------------------------------------------------------------
    if args.dataset_name == "scifact":  # special: concatenate train+test
        qrels = {
            **{
                f"scifact-{k}": v
                for k, v in load_qrels(args.qrels_path).items()
            },
            **{
                f"scifact-{k}": v
                for k, v in load_qrels(args.qrels_path.with_name("train.tsv")).items()
            },
        }
    else:
        qrels = load_qrels(args.qrels_path)

    # ---------------------------------------------------------------------
    # Load runs & perform fusion
    # ---------------------------------------------------------------------
    enc_qid_pid = load_all_result_files(
        args.result_dir / args.dataset_name, args.dataset_name, args.mixgr_merge_method
    )

    if args.ret_merge_method == "normalized_sum":
        fused = normalized_sum_fusion(enc_qid_pid)
    elif args.ret_merge_method == "rrf":
        fused = rrf_fusion(enc_qid_pid)
    else:  # weighted_sum
        if not args.weights_dir:
            parser.error("--weights-dir is required for weighted_sum fusion")
        weights = load_encoder_weights(args.dataset_name, args.weights_dir, args.mixgr_merge_method)
        fused = weighted_sum_fusion(enc_qid_pid, weights)

    # ---------------------------------------------------------------------
    # Evaluate & report
    # ---------------------------------------------------------------------
    ndcg5, ndcg20 = get_metrics(fused, qrels)
    avg5 = sum(ndcg5.values()) / len(ndcg5)
    avg20 = sum(ndcg20.values()) / len(ndcg20)
    logging.info("Average NDCG@5  = %.4f", avg5)
    logging.info("Average NDCG@20 = %.4f", avg20)


if __name__ == "__main__":
    main()