
"""Build an indexed SQLite document store for fast passage lookup after retrieval.

Usage:
  python scripts/build_docstore.py --lang en
  python scripts/build_docstore.py --lang hi
"""
import argparse
import json
import sqlite3
from pathlib import Path
from backend.config import pack_dir

def build(lang):
    pack = pack_dir(lang)
    source = pack / "passages.jsonl"
    target = pack / "passages.sqlite"
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        target.unlink()
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE passages (docid TEXT PRIMARY KEY, title TEXT, text TEXT)")
    batch = []
    count = 0
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            batch.append((str(row.get("id", row.get("docid", ""))),
                          row.get("title", ""), row.get("text", row.get("contents", ""))))
            if len(batch) >= 5000:
                conn.executemany("INSERT OR REPLACE INTO passages VALUES (?, ?, ?)", batch)
                count += len(batch)
                batch.clear()
                print(f"{lang}: indexed {count:,} passages")
    if batch:
        conn.executemany("INSERT OR REPLACE INTO passages VALUES (?, ?, ?)", batch)
        count += len(batch)
    conn.commit()
    conn.close()
    print(f"Finished {lang}: {count:,} passages -> {target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "hi"], required=True)
    args = parser.parse_args()
    build(args.lang)
