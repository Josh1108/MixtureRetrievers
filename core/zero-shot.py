#!/usr/bin/env python3
"""
dynamic_merge.py  (JSONL-only)

Fuse CSV runs with (one or more) weight files and write JSONL results.
Before fusion, **collapse**:
  - subqueries qid 'Q#i' → base query 'Q' (aggregate scores per doc with max/mean)
  - prop doc ids         → base chunk doc ids (aggregate with max/mean)

Inputs (runs): CSV (no header)
  <enc>__<query|subquery>_<chunk|prop>__<tag>.csv
  rows: qid,doc_id,score

Weight files (JSON):
  - Pre weights (from pre-retrieval-weights.py):
      {"type":"pre","weights": { "<qid>": { "<enc|variant|qtype>": float, ... }, ... }}
  - Optional post weights:
      {"type":"post","scope":"global","weights": { "<qid>": {"<doc>": float, ...}, ... }}
      {"type":"post","scope":"per_retriever","weights": { "<qid>": {"<rkey>": {"<doc>": float}}, ... }}

Fusion:
  S_final(q,d) = sum_r [ pre(q,r) * norm_r(q,d) * post_global(q,d) * post_r(q,r,d) ]

Output (JSONL lines; **whole query + chunk doc ids**):
  {"query_id": q, "doc_id": d, "rank": k, "final_score": s, "contrib": {"rkey": part, ...}}
"""
from __future__ import annotations
import argparse, json, math, re, time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict

# ---------- I/O ----------
def _read_json(path: Path): return json.loads(path.read_text())
def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def _now_tag() -> str: return time.strftime("%Y%m%d_%H%M%S")

# ---------- doc-id collapsing ----------
def extract_base_pid(pid: str, dataset_name: str) -> Tuple[str, str]:
    """
    Return (base_pid, kind), where kind ∈ {'chunk','prop'}.
    Rule matches your zero-shot-fusion reference:
      - For nfcorpus: len(parts)>3 → prop -> drop last 2, else chunk -> drop last 1
      - Else:         len(parts)>2 → prop -> drop last 2, else chunk -> drop last 1
    """
    parts = pid.split("-")
    if dataset_name == "nfcorpus":
        if len(parts) > 3:
            return "-".join(parts[:-2]), "prop"
        return "-".join(parts[:-1]), "chunk"
    else:
        if len(parts) > 2:
            return "-".join(parts[:-2]), "prop"
        return "-".join(parts[:-1]), "chunk"

# ---------- runs ----------
def read_csv_triples(path: Path) -> List[dict]:
    """
    CSV rows: qid,doc_id,score (no header).
    Infer retriever key from filename:
      <encsan>__<query|subquery>_(chunk|prop)__<tag>.csv
    retriever_key = encsan|<chunk|prop>|<query|subq>
    """
    m = re.match(r"(.+)__(query|subquery)_(chunk|prop)__(.+)\.csv$", path.name)
    if not m: raise ValueError(f"Bad run filename: {path.name}")
    encsan, qtype, variant, _ = m.groups()
    qtype = "subq" if qtype == "subquery" else "query"
    rkey = f"{encsan}|{variant}|{qtype}"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            qid, did, score = ln.split(",", 3)
            rows.append({"retriever_key": rkey, "qid": qid, "doc_id": did, "score": float(score)})
    return rows

def collect_runs(paths: List[Path]) -> List[dict]:
    all_rows: List[dict] = []
    for p in paths:
        if p.suffix.lower() != ".csv": raise ValueError(f"Only CSV runs supported: {p}")
        all_rows.extend(read_csv_triples(p))
    return all_rows

# ---------- norms ----------
def minmax_norm(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    if not pairs: return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    if math.isclose(lo, hi): return {d: 0.0 for d, _ in pairs}
    rng = hi - lo
    return {d: (s - lo) / rng for d, s in pairs}

def identity_norm(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    # True "none": return raw scores unchanged
    return {d: s for d, s in pairs}

# ---------- weights ----------
def load_weight_files(paths: List[Path]):
    pre_list, post_global_list, post_per_ret_list = [], [], []
    for p in paths:
        obj = _read_json(p)
        w = obj.get("weights", {})
        typ = obj.get("type", "pre").lower()
        scope = obj.get("scope", None)
        if typ == "pre":
            pre_list.append(w)
        elif scope == "per_retriever":
            post_per_ret_list.append(w)
        else:
            post_global_list.append(w)
    return pre_list, post_global_list, post_per_ret_list

# ---------- fuse (with collapse) ----------
def merge_runs(
    runs: List[dict],
    pre_weight_dicts: List[dict],
    post_global_dicts: List[dict],
    post_per_ret_dicts: List[dict],
    dataset_name: str,
    collapse_mode: str = "max",  # how to collapse subq + prop → whole + chunk
    score_norm: str = "minmax",
    final_top_k: int = 1000,
):
    """
    Steps:
      1) Collapse qid 'Q#i' → 'Q' and doc ids prop→chunk using extract_base_pid().
      2) Aggregate duplicates per (retriever, base_qid, base_pid) via max/mean.
      3) Optionally normalize per retriever (none|minmax).
      4) Fuse with pre/post weights.
    """
    # choose normalization
    norm_fn = identity_norm if score_norm == "none" else minmax_norm
    use_mean = (collapse_mode == "mean")

    # Accumulate raw scores while collapsing to base ids
    # stats[(rk, base_qid)][base_pid] = (sum, count, max)
    stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0, float("-inf")]))

    for r in runs:
        rk = r["retriever_key"]
        base_qid = r["qid"].split("#", 1)[0]             # collapse subqueries → whole
        base_pid, _kind = extract_base_pid(r["doc_id"], dataset_name)  # collapse prop → chunk
        s = float(r["score"])
        entry = stats[(rk, base_qid)][base_pid]
        entry[0] += s                   # sum
        entry[1] += 1                   # count
        if s > entry[2]: entry[2] = s   # max

    # Materialize collapsed hit maps (doc -> score) using chosen aggregator
    hits = {}
    for key, doc_stats in stats.items():
        out = {}
        for pid, (ssum, cnt, mmax) in doc_stats.items():
            out[pid] = (ssum / cnt) if use_mean and cnt > 0 else mmax
        hits[key] = out

    retrievers = sorted({rk for rk, _ in hits.keys()})
    parent_qids = sorted({q for _, q in hits.keys()})

    merged_rows = []
    for qid in parent_qids:
        # normalize per retriever (or raw if --score_norm none)
        per_r_norm: Dict[str, Dict[str, float]] = {}
        for rk in retrievers:
            raw_map = hits.get((rk, qid), {})
            if not raw_map: continue
            per_r_norm[rk] = norm_fn(list(raw_map.items()))

        accum = defaultdict(float)                            # doc -> fused score
        contrib = defaultdict(lambda: defaultdict(float))     # doc -> rkey -> part

        for rk, dmap in per_r_norm.items():
            # pre weight: multiply across all pre files; default 0.0 if any pre present but missing; 1.0 if no pre files
            if pre_weight_dicts:
                w_pre, key_found = 1.0, False
                for pd in pre_weight_dicts:
                    w = pd.get(qid, {}).get(rk, None)
                    if w is None: w_pre *= 0.0
                    else: key_found, w_pre = True, (w_pre * float(w))
                if not key_found: w_pre = 0.0
            else:
                w_pre = 1.0
            if w_pre == 0.0: continue

            for doc_id, s_norm in dmap.items():
                # post global(q,d)
                w_pg = 1.0
                for gd in post_global_dicts:
                    w_pg *= float(gd.get(qid, {}).get(doc_id, 1.0))
                # post per retriever(q,r,d)
                w_pr = 1.0
                for rd in post_per_ret_dicts:
                    w_pr *= float(rd.get(qid, {}).get(rk, {}).get(doc_id, 1.0))

                part = w_pre * s_norm * w_pg * w_pr
                if part != 0.0:
                    accum[doc_id] += part
                    contrib[doc_id][rk] = contrib[doc_id].get(rk, 0.0) + part

        ranked = sorted(accum.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            merged_rows.append({
                "query_id": qid,           # whole query id
                "doc_id": doc_id,          # base chunk id
                "rank": rank,
                "final_score": score,
                "contrib": contrib[doc_id]
            })

    return merged_rows, retrievers

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="JSONL-only fusion of runs with (pre/post) weights.")
    ap.add_argument("--runs", required=True, help="Comma-separated CSV run paths.")
    ap.add_argument("--weight_files", required=True, help="Comma-separated JSON weight paths (pre/post).")
    ap.add_argument("--out_dir", required=True, help="Where to write merged_*.jsonl")
    ap.add_argument("--dataset_name", default="scifact", help="Dataset name (affects prop→chunk collapsing).")
    ap.add_argument("--collapse_mode", choices=["max", "mean"], default="max",
                    help="How to aggregate duplicates when collapsing subq/prop to whole/chunk.")
    ap.add_argument("--score_norm", choices=["none", "minmax"], default="minmax",
                    help="Per-retriever per-query doc score normalization.")
    ap.add_argument("--final_top_k", type=int, default=1000)
    args = ap.parse_args()

    run_paths = [Path(p) for p in args.runs.split(",") if p]
    w_paths = [Path(p) for p in args.weight_files.split(",") if p]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = _now_tag()

    runs = collect_runs(run_paths)
    pre_list, post_global_list, post_per_ret_list = load_weight_files(w_paths)

    merged_rows, retrievers = merge_runs(
        runs, pre_list, post_global_list, post_per_ret_list,
        dataset_name=args.dataset_name,
        collapse_mode=args.collapse_mode,
        score_norm=args.score_norm,
        final_top_k=args.final_top_k
    )

    out_jsonl = out_dir / f"merged_{tag}.jsonl"
    _write_jsonl(out_jsonl, merged_rows)
    print(json.dumps({
        "merged_jsonl": str(out_jsonl),
        "num_queries": len({x['query_id'] for x in merged_rows}),
        "retrievers": retrievers
    }, indent=2))

if __name__ == "__main__":
    main()
