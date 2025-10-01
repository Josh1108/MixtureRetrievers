from core.indexing import MultiIndexRetriever

encoders = [
    "bm25",
    "SimCSE", "Contriever", "DPR", "ANCE", 
    "GTR-T5-base", 
    "MPNet",
]

engine = MultiIndexRetriever(
    corpus_path="/home/jkalra/final_repo_directed_Study/MixtureRetrievers/data/nfcorpus/corpus",
    output_dir="/data/user_data/jkalra/indexes",
    encoders=encoders,
    index_type="flat",
    batch_size=128,
    top_k=100,                                # top-k per encoder per variant
    build_props=True,
)

engine.ensure_all_indexes()

# If you already have propositionized subqueries, pass subqueries="PATH/TO/queries.multi.jsonl"
out = engine.search_and_dump_4way_csv_ids(
    queries="/home/jkalra/final_repo_directed_Study/MixtureRetrievers/data/nfcorpus/queries/queries.whole.jsonl",    # rows: {"id": "...", "title": "..."}
    subqueries="/home/jkalra/final_repo_directed_Study/MixtureRetrievers/data/nfcorpus/queries/queries.multi.jsonl",
    top_k=100,
    run_tag_prefix=None,                      # or set your own tag like "expA_v1"
)

print("✅ Wrote CSV runs:")
for p in out["files"]:
    print("  -", p)