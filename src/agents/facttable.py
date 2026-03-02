from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from .chunker import ChunkingEngine
from ..models.schemas import LDU


DB_PATH = Path('.refinery/facttable.db')


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            key TEXT,
            value TEXT,
            page INTEGER,
            bbox TEXT,
            content_hash TEXT
        )
        '''
    )
    conn.commit()
    return conn


def extract_facts_from_ldus(ldus: List[LDU]) -> int:
    """Naive fact extractor: finds numeric patterns with surrounding header-like tokens and stores them.
    Returns number of facts inserted."""
    conn = _ensure_db()
    cur = conn.cursor()
    num = 0
    num_pattern = re.compile(r"\$?\d{1,3}(?:[\,\.]\d{3})*(?:[\.,]\d+)?")
    for l in ldus:
        # consider table chunks first
        if l.chunk_type in ("table", "paragraph"):
            matches = num_pattern.findall(l.content)
            if matches:
                # heuristically choose a key from start of content
                key = l.content.strip().split('\n')[0][:80]
                for m in matches:
                    cur.execute(
                        "INSERT INTO facts (doc_id, key, value, page, bbox, content_hash) VALUES (?, ?, ?, ?, ?, ?)",
                        (getattr(l, 'content_hash', ''), key, m, l.page_refs[0] if l.page_refs else None, str(getattr(l, 'bbox', '')), l.content_hash),
                    )
                    num += 1
    conn.commit()
    conn.close()
    return num


def query_facts(sql: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


if __name__ == '__main__':
    print('facttable module')
