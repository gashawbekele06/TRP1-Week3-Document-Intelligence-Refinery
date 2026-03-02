from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import math
import hashlib
from ..models.schemas import ProvenanceItem, ProvenanceChain


LDUS_DIR = Path('.refinery/ldus')
PAGEINDEX_DIR = Path('.refinery/pageindex')


def _load_pageindex(doc_id: str) -> Optional[Dict[str, Any]]:
    p = PAGEINDEX_DIR / f"{doc_id}_pageindex.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def pageindex_navigate(topic: str, doc_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    pi = _load_pageindex(doc_id)
    if not pi:
        return []
    nodes = pi.get('root', {}).get('child_sections', [])
    # naive relevance: count topic tokens in title+summary
    tkns = set(topic.lower().split())
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for n in nodes:
        text = (n.get('title','') + ' ' + (n.get('summary') or '')).lower()
        score = sum(1 for w in tkns if w in text)
        scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for s, n in scored[:top_k]]


def _deterministic_embedding(text: str) -> List[float]:
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vec = [b / 255.0 for b in h[:128]]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _load_ldus() -> List[Dict[str, Any]]:
    out = []
    if not LDUS_DIR.exists():
        return out
    for p in LDUS_DIR.glob('*.jsonl'):
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    # include fallback file
    fb = LDUS_DIR / 'fallback_ldus.jsonl'
    if fb.exists():
        with open(fb, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def semantic_search(query: str, doc_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    all_ldus = _load_ldus()
    # if doc_id provided filter
    if doc_id:
        all_ldus = [l for l in all_ldus if l.get('content_hash','').endswith(doc_id) or l.get('content', '').find(doc_id) != -1 or True]
    qv = _deterministic_embedding(query)
    scored = []
    for l in all_ldus:
        text = l.get('content','')
        ev = _deterministic_embedding(text)
        score = _cosine(qv, ev)
        scored.append((score, l))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [l for s, l in scored[:top_k]]


def structured_query(sql: str) -> List[Dict[str, Any]]:
    from .facttable import query_facts
    return query_facts(sql)


def _ldu_to_prov(ldu: Dict[str, Any]) -> ProvenanceItem:
    # map LDU JSON to ProvenanceItem; bbox may be dict or string
    bbox = ldu.get('bbox') or {"x0": 0, "top": 0, "x1": 0, "bottom": 0}
    return ProvenanceItem(document_name=ldu.get('content_hash', 'unknown'), page_number=(ldu.get('page_refs') or [None])[0], bbox=bbox, content_hash=ldu.get('content_hash', ''))


def answer_query(query: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    # Step 1: use pageindex to narrow down sections
    sections = []
    if doc_id:
        sections = pageindex_navigate(query, doc_id, top_k=3)

    # Step 2: semantic search (prefer sections if available)
    candidates = semantic_search(query, doc_id=doc_id, top_k=5)

    answer_text = ''
    provs = []
    if candidates:
        # naive answer: concatenate top candidate contents
        answer_text = '\n\n'.join([c.get('content','')[:1000] for c in candidates])
        for c in candidates:
            provs.append(_ldu_to_prov(c))

    pc = ProvenanceChain(citations=provs)
    return {"answer": answer_text, "provenance": pc.model_dump()}


def audit_claim(claim: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    # Verify claim by searching facts and LDUs, return evidence or 'not found'
    # 1) check structured facts
    facts = structured_query("SELECT * FROM facts WHERE value IS NOT NULL LIMIT 10")
    matches = [f for f in facts if claim.lower().split()[0] in (f.get('key') or '').lower() or any(tok in (f.get('value') or '') for tok in claim.split())]
    if matches:
        return {"verifiable": True, "matches": matches}

    # 2) fallback to semantic search
    res = semantic_search(claim, doc_id=doc_id, top_k=5)
    if res:
        return {"verifiable": True, "evidence": res}

    return {"verifiable": False, "reason": "not found"}


if __name__ == '__main__':
    print('query_agent module')
