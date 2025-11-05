#!/usr/bin/env python3
import json, time, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import logging

def _logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("mixgr")

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def _read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
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
    Filename: <encsan>__<query|subquery>_<chunk|prop>__<tag>.csv
    -> retriever_key = encsan|<chunk|prop>|<query|subq>
    """
    m = re.match(r"(.+)__(query|subquery)_(chunk|prop)__(.+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Run filename not recognized: {path.name}")
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

def load_queries_whole(path: Path) -> Dict[str, str]:
    """Return {qid: title} from queries.whole.jsonl."""
    out = {}
    for row in _read_jsonl(path):
        qid = str(row.get("id"))
        title = (row.get("title") or "").strip()
        if qid and title:
            out[qid] = title
    if not out:
        raise ValueError(f"No queries loaded from {path}")
    return out

def load_subqueries_multi(path: Path) -> Dict[str, List[str]]:
    """Return {parent_qid: [subquery_text,...]} from queries.multi.jsonl (ids like 'qid#i')."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in _read_jsonl(path):
        sid = str(row.get("id"))
        title = (row.get("title") or "").strip()
        if not sid or not title:
            continue
        parent = sid.split("#", 1)[0]
        groups[parent].append(title)
    return groups
