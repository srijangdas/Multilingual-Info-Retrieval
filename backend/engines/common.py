import json
import sqlite3
from pathlib import Path
from typing import Iterable

def fetch_passages(pack: Path, docids: Iterable[str]) -> dict:
    wanted = [str(x) for x in docids]
    if not wanted:
        return {}
    db = pack / "passages.sqlite"
    if db.exists():
        out = {}
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:

            for start in range(0, len(wanted), 800):
                chunk = wanted[start:start+800]
                marks = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT docid, title, text FROM passages WHERE docid IN ({marks})", chunk
                ).fetchall()
                out.update({str(i): {"title": t or "", "text": x or ""} for i, t, x in rows})
        return out

    source = pack / "passages.jsonl"
    wanted_set, found = set(wanted), {}
    if not source.exists():
        return found
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docid = str(row.get("id", row.get("docid", "")))
            if docid in wanted_set:
                found[docid] = {"title": row.get("title", ""),
                                "text": row.get("text", row.get("contents", ""))}
                if len(found) == len(wanted_set):
                    break
    return found

def attach_passages(pack: Path, docids: list, scores: list, extras=None):
    passages = fetch_passages(pack, docids)
    rows = []
    for rank, (docid, score) in enumerate(zip(docids, scores), 1):
        p = passages.get(str(docid), {})
        row = {"rank": rank, "docid": str(docid), "score": float(score),
               "title": p.get("title", ""), "text": p.get("text", "")}
        if extras and str(docid) in extras:
            row.update(extras[str(docid)])
        rows.append(row)
    return rows
