#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from core.indexing import MultiIndexRetriever  # _sanitize for override keys
from core.indexing import _sanitize

DATASET = "scifact"

def main():
    ap = argparse.ArgumentParser()
    # Point to the FOLDER containing corpus.chunk.jsonl (and maybe corpus.prop.jsonl)
    ap.add_argument("--corpus_dir", default=f"data/{DATASET}/corpus", help="Folder with corpus.chunk.jsonl (and optional corpus.prop.jsonl)")
    ap.add_argument("--queries", default=f"data/{DATASET}/query/queries.whole.jsonl", help="Queries .jsonl with {id,title}")
    ap.add_argument("--output_root", default="/data/user_data/jkalra/indexes", help="Where indexes & runs are stored")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--index_type", choices=["flat","ivf","hnsw"], default="flat")
    ap.add_argument("--tag", default=DATASET, help="Tag for output filenames")
    ap.add_argument("--with_bm25", action="store_true", help="Also run BM25 (optional)")
    ap.add_argument("--run_weights", action="store_true", help="Run thrust-weight merger after retrieval")
    ap.add_argument("--weights_script", default="core/pre-retrieval-weights.py", help="Path to weight/merge script")
    args = ap.parse_args()
    script_dir = Path(__file__).parent
    weights_script_path = script_dir / args.weights_script
    dataset_dir = Path(args.output_root) / DATASET
    run_dir = dataset_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- encoders to run
    encoders: List[str] = [
        "all-mpnet-base-v2",                    # MPNet base v2
        "facebook-dpr-ctx_encoder-multiset-base"  # DPR ctx (SBERT-ready id)
    ]
    if args.with_bm25:
        encoders = ["bm25"] + encoders

    # ---- build/load indexes; auto-uses corpus.chunk.jsonl (+ builds/loads corpus.prop.jsonl)
    engine = MultiIndexRetriever(
        corpus_path=args.corpus_dir,            # FOLDER path
        output_dir=args.output_root,
        dataset_name=DATASET,
        encoders=encoders,
        index_type=args.index_type,
        batch_size=args.batch_size,
        top_k=args.top_k,
        build_props=True,                       # load corpus.prop.jsonl if present; else create it
    )

    print(">> ensuring indexes …")
    engine.ensure_all_indexes()

    query_dir = Path(args.corpus_dir) / "query"
    whole_path = Path(args.queries)
    multi_path = None

    # If user didn't override --queries, try default scifact layout
    if not whole_path.exists():
        cand = query_dir / "queries.whole.jsonl"
        if cand.exists():
            whole_path = cand
        else:
            raise FileNotFoundError(f"Whole queries not found at {whole_path} or {cand}")

    # Optional multi file
    cand_multi = query_dir / "queries.multi.jsonl"
    if cand_multi.exists():
        multi_path = cand_multi
    
    print(">> running retrieval across all encoders (CSV qid,doc_id,score) …")
    out = engine.search_and_dump_4way_csv_ids(
        queries=args.queries,
        subqueries=str(multi_path) if multi_path else None,
        top_k=args.top_k,
        run_tag_prefix=args.tag,
    )
    csv_paths = out["files"]
    print(f">> wrote {len(csv_paths)} CSV files under {run_dir}")
    for p in csv_paths:
        print("  -", p)

    if not args.run_weights:
        print("Done. (Skip weights/merge; pass --run_weights to run it.)")
        return

    # ---- encoder overrides for thrust weights (query/doc encoders)
    overrides: Dict[str, Dict[str, str]] = {}
    if args.with_bm25:
        overrides["bm25"] = {
            "query_model": "all-mpnet-base-v2",
            "doc_model": "all-mpnet-base-v2",
        }
    overrides[_sanitize("all-mpnet-base-v2")] = {
        "query_model": "all-mpnet-base-v2",
        "doc_model": "all-mpnet-base-v2",
    }
    overrides[_sanitize("facebook-dpr-ctx_encoder-multiset-base")] = {
        "query_model": "facebook/dpr-question_encoder-multiset-base",
        "doc_model": "facebook/dpr-ctx_encoder-multiset-base",
    }

    merges_dir = dataset_dir / "merges"
    merges_dir.mkdir(parents=True, exist_ok=True)
    overrides_path = merges_dir / f"encoder_overrides.{args.tag}.json"
    overrides_path.write_text(json.dumps(overrides, indent=2))
    print(">> wrote encoder overrides:", overrides_path)

    # ---- call weights/merge script (expects CSV runs + queries)
    runs_arg = ",".join(csv_paths)
    
    cmd = [
        sys.executable, str(weights_script_path),
        "--runs", runs_arg,
        "--queries", str(args.queries),            # needed to embed queries/subqueries
        "--dataset_dir", str(dataset_dir),
        "--out_dir", str(merges_dir),
        "--encoder_overrides", str(overrides_path),
        "--default_embedder", "all-mpnet-base-v2",
        "--top_k", str(args.top_k),
        "--weight_norm", "softmax",
        "--softmax_temp", "1.0",
    ]
    print(">> calling:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("All done.")

if __name__ == "__main__":
    main()
