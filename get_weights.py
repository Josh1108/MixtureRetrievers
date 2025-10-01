from pathlib import Path
from core.pre_retrieval_weights import compute_pre_weights

# --- EDIT THESE ---
DATASET_DIR = Path("/data/user_data/jkalra/indexes/nfcorpus")                     # e.g. indexes/scifact
RUNS_DIR    = DATASET_DIR / "runs"
OUT_DIR     = DATASET_DIR / "weights"
QUERIES     = Path("/home/jkalra/final_repo_directed_Study/MixtureRetrievers/data/nfcorpus/queries/queries.whole.jsonl")
SUBQUERIES  = Path("/home/jkalra/final_repo_directed_Study/MixtureRetrievers/data/nfcorpus/queries/queries.multi.jsonl")                                       # optional, leave blank if not used
DEFAULT_EMB = "na"
TRAIN_SAMPLE = None                                          # e.g., 5000
ENC_OVERRIDES = None                                         # e.g., Path("enc_overrides.yaml")
# ------------------

runs = sorted(RUNS_DIR.glob("*.csv"))
if not runs:
    raise SystemExit(f"No CSV runs found in {RUNS_DIR}")

weights_path = compute_pre_weights(
    runs=runs,
    queries=QUERIES,
    dataset_dir=DATASET_DIR,
    out_dir=OUT_DIR,
    default_embedder=DEFAULT_EMB,
    subqueries=(SUBQUERIES if SUBQUERIES and SUBQUERIES.exists() else None),
    encoder_overrides=ENC_OVERRIDES,
    batch_size=128,
    train_sample=TRAIN_SAMPLE,
)
print(f"✅ Wrote: {weights_path}")
