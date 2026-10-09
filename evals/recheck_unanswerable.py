"""Re-run the answerability check on every near-miss question flagged partial/unparsed, with room for a full reply.

Writes evals/results/recheck.json: id -> {verdict, evidence, passage} for the reviewer to read. Free of retrieval cost
(local), about 10 cents of Sonnet calls. Run: uv run python -m evals.recheck_unanswerable
"""

import json
import re
from pathlib import Path

import psycopg

from app.config import get_settings
from app.embeddings import get_embedder
from app.llm import complete
from app.retrieval.rerank import get_reranker
from evals.draft_more import ANSWERABLE
from evals.run import _ranker, scopes_from_db
from evals.schema import EvalItem


def main() -> None:
    s = get_settings()
    items = [EvalItem.model_validate_json(line) for line in open("evals/qa_new_draft.jsonl")]
    todo = [
        i
        for i in items
        if i.kind == "unanswerable" and any(f.startswith("checker_partial") for f in i.flags)
    ]
    conn = psycopg.connect(s.database_url("owner"), autocommit=True)
    rank = _ranker("hybrid+rerank", conn, get_embedder(s), get_reranker(s), scopes_from_db(conn), s)
    out = {}
    for it in todo:
        top = rank(it)[:8]
        ctx = "\n\n".join(
            f"[{n}] {c.section_path}\n{' '.join(c.text.split())[:900]}"
            for n, c in enumerate(top, 1)
        )
        r = complete([{"role": "user", "content": ANSWERABLE.format(q=it.question, ctx=ctx)}], name="answerability",
                     model=s.llm_judge_model, max_tokens=500, temperature=None)  # fmt: skip
        text = re.sub(r"^```(?:json)?|```$", "", r.text.strip(), flags=re.M).strip()
        try:
            v = json.loads(text)
        except json.JSONDecodeError:
            v = {"verdict": "unparsed", "evidence": text[:300]}
        out[it.id] = {"question": it.question, **v}
        print(f"{it.id}: {v.get('verdict')} | {str(v.get('evidence'))[:200]}", flush=True)
    Path("evals/results/recheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
