
from pathlib import Path
from typing import List, Dict, Optional, Literal, Any
import tempfile
import json
import shutil
import numpy as np

import importlib.util

from core.indexing import MultiIndexRetriever, _sanitize
from core.pre_retrieval_weights import compute_pre_weights
from core.utils import _read_jsonl, _write_json

_zero_shot_spec = importlib.util.spec_from_file_location(
    "zero_shot", Path(__file__).parent / "core" / "zero-shot.py"
)
_zero_shot = importlib.util.module_from_spec(_zero_shot_spec)
_zero_shot_spec.loader.exec_module(_zero_shot)
merge_runs = _zero_shot.merge_runs
collect_runs = _zero_shot.collect_runs
load_weight_files = _zero_shot.load_weight_files


class MixtureRetriever:
    """
    Simple interface for running retrieval on custom queries and documents.
    
    This class handles all the complexity of:
    - Building indexes from documents
    - Running multiple retrieval methods
    - Computing and applying pre-retrieval weights
    - Merging results from different retrievers
    - Returning ranked, scored results
    """
    
    def __init__(
        self,
        retrievers: Optional[List[str]] = None,
        use_pre_weights: bool = True,
        use_post_weights: bool = False,
        pre_weight_threshold: Optional[float] = None,
        weight_norm: Literal["none", "softmax", "minmax"] = "softmax",
        softmax_temp: float = 1.0,
        index_type: str = "flat",
        batch_size: int = 128,
        build_props: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the MixtureRetriever.
        
        Args:
            retrievers: List of encoder names (e.g., ["all-mpnet-base-v2", "bm25"]).
                      Default: ["all-mpnet-base-v2", "facebook-dpr-ctx_encoder-multiset-base"]
            use_pre_weights: Whether to use pre-retrieval weights (thrust/KB scores).
                            When True, computes weights per query/retriever and uses them for fusion.
            use_post_weights: Whether to use post-retrieval weights (currently not implemented).
            pre_weight_threshold: If set, filters out retrievers with pre-weight below this threshold.
                                 After filtering, weights are re-normalized.
            weight_norm: How to normalize weights:
                        - "none": Use raw weights
                        - "softmax": Apply softmax normalization
                        - "minmax": Min-max normalization to [0, 1]
            softmax_temp: Temperature for softmax normalization (higher = more uniform).
            index_type: FAISS index type ("flat", "ivf", "hnsw"). "flat" is fastest for small datasets.
            batch_size: Batch size for encoding documents and queries.
            build_props: Whether to build proposition-level indexes (slower but sometimes better).
            verbose: Whether to print progress messages.
        """
        self.retrievers = retrievers or [
            "all-mpnet-base-v2",
            "facebook-dpr-ctx_encoder-multiset-base"
        ]
        self.use_pre_weights = use_pre_weights
        self.use_post_weights = use_post_weights
        self.pre_weight_threshold = pre_weight_threshold
        self.weight_norm = weight_norm
        self.softmax_temp = softmax_temp
        self.index_type = index_type
        self.batch_size = batch_size
        self.build_props = build_props
        self.verbose = verbose
        
        # Temporary directory for this session
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mor_"))
        self.corpus_dir = self.temp_dir / "corpus"
        self.index_dir = self.temp_dir / "indexes"
        self.runs_dir = self.temp_dir / "runs"
        self.weights_dir = self.temp_dir / "weights"
        
        # Create directories
        for d in [self.corpus_dir, self.index_dir, self.runs_dir, self.weights_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.engine = None
        self._corpus_name = "custom"
        self._doc_map = {}
    
    def _log(self, message: str):
        """Print message if verbose is enabled."""
        if self.verbose:
            print(message)
    
    def _prepare_corpus(self, documents: List[str]) -> Path:
        """Convert list of documents to corpus.chunk.jsonl format."""
        corpus_path = self.corpus_dir / "corpus.chunk.jsonl"
        
        self._log(f"Preparing corpus from {len(documents)} documents...")
        
        with corpus_path.open("w", encoding="utf-8") as f:
            for i, doc in enumerate(documents):
                doc_id = f"doc_{i}"
                doc_text = doc.strip()
                if not doc_text:
                    continue
                f.write(json.dumps({
                    "id": doc_id,
                    "contents": doc_text
                }, ensure_ascii=False) + "\n")
                self._doc_map[doc_id] = doc_text
        
        return self.corpus_dir
    
    def _prepare_queries(self, queries: List[str]) -> Path:
        """Convert list of queries to queries.whole.jsonl format."""
        query_path = self.corpus_dir / "queries" / "queries.whole.jsonl"
        query_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Preparing {len(queries)} queries...")
        
        with query_path.open("w", encoding="utf-8") as f:
            for i, query in enumerate(queries):
                f.write(json.dumps({
                    "id": f"Q{i}",
                    "title": query.strip()
                }, ensure_ascii=False) + "\n")
        
        return query_path
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights using specified method."""
        if not weights:
            return weights
        
        if self.weight_norm == "none":
            return weights
        elif self.weight_norm == "softmax":
            keys = list(weights.keys())
            values = np.array([weights[k] for k in keys])
            # Avoid numerical issues
            if np.max(values) == np.min(values):
                return {k: 1.0 / len(keys) for k in keys}
            # Apply temperature
            values = values / self.softmax_temp
            # Softmax
            exp_vals = np.exp(values - np.max(values))
            normalized = exp_vals / exp_vals.sum()
            return {k: float(v) for k, v in zip(keys, normalized)}
        elif self.weight_norm == "minmax":
            values = list(weights.values())
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 1.0 / len(weights) for k in weights.keys()}
            return {k: (v - min_val) / (max_val - min_val) for k, v in weights.items()}
        return weights
    
    def _apply_threshold(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply threshold to filter out low-weight retrievers."""
        if self.pre_weight_threshold is None:
            return weights
        
        filtered = {
            k: v for k, v in weights.items() 
            if v >= self.pre_weight_threshold
        }
        
        if not filtered:
            # If all were filtered, keep the original (don't filter everything)
            self._log(f"Warning: All retrievers filtered by threshold {self.pre_weight_threshold}, keeping original weights")
            return weights
        
        # Re-normalize if any were filtered
        if len(filtered) < len(weights):
            self._log(f"Filtered {len(weights) - len(filtered)} retrievers below threshold {self.pre_weight_threshold}")
            return self._normalize_weights(filtered)
        
        return weights
    
    def _process_weights(self, weights_path: Path) -> Path:
        """Load weights, normalize, and apply threshold."""
        weights_data = json.loads(weights_path.read_text())
        raw_weights = weights_data["weights"]
        
        # Process weights per query
        processed_weights = {}
        for qid, q_weights in raw_weights.items():
            normalized = self._normalize_weights(q_weights)
            thresholded = self._apply_threshold(normalized)
            processed_weights[qid] = thresholded
        
        # Update weights file
        weights_data["weights"] = processed_weights
        weights_data["meta"]["normalization"] = self.weight_norm
        weights_data["meta"]["threshold"] = self.pre_weight_threshold
        weights_path.write_text(json.dumps(weights_data, indent=2))
        
        return weights_path
    
    def search(
        self,
        queries: List[str],
        documents: List[str],
        top_k: int = 10,
        return_scores: bool = True,
        return_texts: bool = True
    ) -> List[Dict]:
        """
        Search documents for queries using mixture of retrievers.
        
        Args:
            queries: List of query strings to search for.
            documents: List of document strings to search in.
            top_k: Number of top documents to return per query.
            return_scores: Whether to include scores in results.
            return_texts: Whether to include document texts in results.
        
        Returns:
            List of results, one dict per query:
            [
                {
                    "query": str,              # Original query text
                    "query_id": str,           # Internal query ID
                    "results": [
                        {
                            "doc_id": str,     # Document ID
                            "rank": int,       # Rank (1-indexed)
                            "score": float,    # Final fusion score (if return_scores=True)
                            "text": str         # Document text (if return_texts=True)
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        if not queries:
            raise ValueError("queries list cannot be empty")
        if not documents:
            raise ValueError("documents list cannot be empty")
        
        # Prepare corpus and queries
        corpus_path = self._prepare_corpus(documents)
        queries_path = self._prepare_queries(queries)
        
        # Initialize retriever engine
        self._log(f"Initializing retriever with {len(self.retrievers)} methods: {self.retrievers}")
        
        self.engine = MultiIndexRetriever(
            corpus_path=str(corpus_path),
            output_dir=str(self.index_dir),
            dataset_name=self._corpus_name,
            encoders=self.retrievers,
            index_type=self.index_type,
            batch_size=self.batch_size,
            top_k=top_k,
            build_props=self.build_props,
        )
        
        # Build indexes
        self._log("Building indexes (this may take a while for first run)...")
        self.engine.ensure_all_indexes()
        
        # Run retrieval
        self._log("Running retrieval across all methods...")
        output = self.engine.search_and_dump_4way_csv_ids(
            queries=str(queries_path),
            top_k=top_k,
            run_tag_prefix="custom",
        )
        csv_paths = [Path(p) for p in output["files"]]
        
        # Compute pre-weights if requested
        weight_files = []
        if self.use_pre_weights:
            self._log("Computing pre-retrieval weights...")
            weights_path = compute_pre_weights(
                runs=csv_paths,
                queries=queries_path,
                dataset_dir=self.index_dir / self._corpus_name,
                out_dir=self.weights_dir,
                default_embedder="all-mpnet-base-v2",
                subqueries=None,
                encoder_overrides=None,
                batch_size=self.batch_size,
                train_sample=None,
            )
            
            # Process weights (normalize and threshold)
            weights_path = self._process_weights(weights_path)
            weight_files.append(weights_path)
        
        # Merge runs with weights
        self._log("Merging results from all retrievers...")
        
        runs_data = collect_runs(csv_paths)
        pre_list, post_global_list, post_per_ret_list = load_weight_files(weight_files)
        
        merged_rows, _ = merge_runs(
            runs=runs_data,
            pre_weight_dicts=pre_list,
            post_global_dicts=post_global_list,
            post_per_ret_dicts=post_per_ret_list,
            dataset_name=self._corpus_name,
            collapse_mode="max",
            score_norm="minmax",
            final_top_k=top_k,
        )
        
        # Format results
        query_map = {q["id"]: q["title"] for q in self.engine._load_queries(queries_path)}
        
        results = []
        for qid in sorted(query_map.keys()):
            query_results = [
                r for r in merged_rows 
                if r["query_id"] == qid
            ]
            query_results.sort(key=lambda x: x["rank"])
            
            formatted_results = []
            for r in query_results[:top_k]:
                result = {
                    "doc_id": r["doc_id"],
                    "rank": r["rank"],
                }
                if return_scores:
                    result["score"] = float(r["final_score"])
                if return_texts:
                    result["text"] = self._doc_map.get(r["doc_id"], "")
                
                formatted_results.append(result)
            
            results.append({
                "query": query_map[qid],
                "query_id": qid,
                "results": formatted_results
            })
        
        self._log(f"Search complete! Found results for {len(results)} queries.")
        
        return results
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self._log("Cleaned up temporary files")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto cleanup."""
        self.cleanup()


if __name__ == "__main__":
    # Example usage
    retriever = MixtureRetriever(
        retrievers=["all-mpnet-base-v2", "bm25"],
        use_pre_weights=True,
        pre_weight_threshold=0.1,
        weight_norm="softmax",
        verbose=True
    )
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning"
    ]
    
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        "Neural networks are computational models inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers of neural networks to learn complex patterns in data.",
        "Supervised learning requires labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds patterns and structures in data without labeled examples.",
        "Reinforcement learning involves an agent learning to make decisions through trial and error.",
    ]
    
    try:
        results = retriever.search(
            queries=queries,
            documents=documents,
            top_k=3,
            return_scores=True,
            return_texts=True
        )
        
        # Print results
        for result in results:
            print(f"\n{'='*60}")
            print(f"Query: {result['query']}")
            print(f"Found {len(result['results'])} results:")
            for r in result['results']:
                print(f"  [{r['rank']}] (score: {r['score']:.4f})")
                print(f"      {r['text'][:150]}...")
    finally:
        retriever.cleanup()

