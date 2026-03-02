from pathlib import Path
from src.agents.triage import classify_document
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import answer_query, audit_claim
from src.agents.facttable import extract_facts_from_ldus
import json

DATA_DIR = Path('data')
OUT_QA = Path('.refinery/qa_examples.jsonl')
OUT_AUDIT = Path('.refinery/audit.jsonl')


def pick_topic_from_pageindex(pi_path: Path):
    try:
        pi = json.loads(pi_path.read_text(encoding='utf-8'))
        nodes = pi.get('root', {}).get('child_sections', [])
        for n in nodes:
            if n.get('key_entities'):
                return n.get('key_entities')[0]
            if n.get('title') and len(n.get('title')) > 3:
                return n.get('title').split()[0]
    except Exception:
        pass
    return 'revenue'


def process_files(n=12):
    files = sorted(list(DATA_DIR.glob('*.pdf')))[:n]
    OUT_QA.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)

    for f in files:
        print('Processing', f.name)
        profile = classify_document(str(f))
        router = ExtractionRouter(str(f), profile.model_dump())
        res = router.run()
        doc = res.get('document')
        chunker = ChunkingEngine()
        ldus = chunker.chunk_document(doc)
        # persist ldus
        out_ldus = Path('.refinery/ldus')
        out_ldus.mkdir(parents=True, exist_ok=True)
        chunker.emit_ldus_jsonl(ldus, out_ldus / f"{doc.doc_id}_ldus.jsonl")
        # pageindex
        builder = PageIndexBuilder()
        pi = builder.build(doc, ldus)
        builder.persist(pi, Path('.refinery/pageindex'))
        # ingest
        builder.ingest_ldus(ldus)
        # facts
        numfacts = extract_facts_from_ldus(ldus)
        print('  facts:', numfacts)
        # QA: one query per doc based on pageindex
        pi_path = Path('.refinery/pageindex') / f"{doc.doc_id}_pageindex.json"
        topic = pick_topic_from_pageindex(pi_path)
        q = f"Find information about {topic}"
        ans = answer_query(q, doc_id=doc.doc_id)
        qa = {"doc": doc.doc_id, "question": q, "answer": ans.get('answer'), "provenance": ans.get('provenance')}
        with open(OUT_QA, 'a', encoding='utf-8') as fqa:
            fqa.write(json.dumps(qa) + '\n')
        # audit: try a simple claim verification for 'revenue' or the topic
        audit = audit_claim(topic, doc_id=doc.doc_id)
        aud = {"doc": doc.doc_id, "claim": topic, "audit": audit}
        with open(OUT_AUDIT, 'a', encoding='utf-8') as fa:
            fa.write(json.dumps(aud) + '\n')

    print('done')


if __name__ == '__main__':
    process_files(12)
