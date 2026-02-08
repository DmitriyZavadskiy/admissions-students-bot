import argparse
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CMDS = {
    "parse_dcs": ("parse_dcs.py", "DocsParser"),
    "chunks": ("chunks.py", "ChunkMaker"),
    "indexes_for_Qadr": ("indexes_for_Qadr.py", "QdrantIndexer"),
    "test_search": ("test_search.py", "SmokeSearch"),
    "retrieval": ("retrieval.py", "RetrievalEval"),
    "rag": ("rag.py", "RagLocalChat"),
}


def run(cmd: str) -> None:
    fn, cn = CMDS[cmd]
    fp = ROOT / fn
    ns = runpy.run_path(str(fp))
    cls = ns[cn]
    cls().run()


prs = argparse.ArgumentParser(prog="python -m scripts", add_help=True)
prs.add_argument("cmd", choices=sorted(CMDS), help="какой скрипт запустить")
arg = prs.parse_args()
run(arg.cmd)
