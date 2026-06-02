import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CMDS = {
    "parse_dcs": ("parse_dcs.py", "DocsParser"),
    "chunks": ("chunks.py", "ChunkMaker"),
    "indexes_for_Qadr": ("indexes_for_Qadr.py", "QdrantIndexer"),
    "test_search": ("test_search.py", "SmokeSearch"),
    "retrieval": ("retrieval.py", "RetrievalEval"),
    "rag": ("rag.py", "RagLocalChat"),
    "eval_retrieval": ("eval_retrieval.py", "main"),
    "enrich_eval": ("enrich_eval_dataset.py", "main"),
    "extend_eval": ("extend_eval_dataset.py", "main"),
    "chunk_experiments": ("chunk_experiments.py", "main"),
    "retrieval_experiments": ("retrieval_experiments.py", "main"),
    "preprocessing_experiments": ("preprocessing_experiments.py", "main"),
    "reranker_grid": ("reranker_grid.py", "main"),
    "category_analysis": ("category_analysis.py", "main"),
    "generation_eval": ("generation_eval.py", "main"),
    "generation_review": ("generation_review.py", "main"),
    "enrich_chunks": ("enrich_chunks.py", "main"),
    "eval_enriched": ("eval_enriched.py", "main"),
    "gen_enriched": ("gen_enriched.py", "main"),
    "reranker_latency": ("reranker_latency.py", "main"),
}


def run(cmd: str, args: list[str]) -> None:
    fn, cn = CMDS[cmd]
    fp = ROOT / fn
    ns = runpy.run_path(str(fp))
    obj = ns[cn]
    sys.argv = [f"python -m scripts {cmd}", *args]
    if cn == "main":
        obj()
    else:
        obj().run()


prs = argparse.ArgumentParser(prog="python -m scripts", add_help=True)
prs.add_argument("cmd", choices=sorted(CMDS), help="какой скрипт запустить")
arg = prs.parse_args(sys.argv[1:2])
rest = sys.argv[2:]
run(arg.cmd, rest)
