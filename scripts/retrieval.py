import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLL = "admissions_chunks"
EMBMDL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

QUESTIONS = Path("data/eval/gold_qa.json")
TOPK = 5

URL = "http://localhost:6333"

class RetrievalEval:
    def __init__(
        self,
        coll: str = COLL,
        mdl: str = EMBMDL,
        questions: Path = QUESTIONS,
        topk: int = TOPK,
        url: str = URL,
    ):
        self.coll = coll
        self.mdl = mdl
        self.questions = questions
        self.topk = topk
        self.url = url

    @staticmethod
    def norm_nm(txt: str) -> str:
        return (txt or "").lower()

    @staticmethod
    def rd(pth: Path):
        with pth.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _qry(self, qdr: QdrantClient, emb: SentenceTransformer, que: str, lim: int):
        qvec = emb.encode(que, normalize_embeddings=True).tolist()
        res = qdr.query_points(
            collection_name=self.coll,
            query=qvec,
            limit=lim,
            with_payload=True,
        )
        return res.points if hasattr(res, "points") else res

    def run(self) -> None:
        qdr = QdrantClient(url=self.url)
        emb = SentenceTransformer(self.mdl)

        tot = 0
        h1 = 0
        hk = 0

        for ex in self.rd(self.questions):
            que = ex["question"]
            exp = self.norm_nm(ex.get("expected_doc", ""))

            hits = self._qry(qdr, emb, que, lim=self.topk)
            tts: list[str] = []
            for hit in hits:
                pay = getattr(hit, "payload", None) or {}
                tit = self.norm_nm(pay.get("title", ""))
                src = self.norm_nm(pay.get("source", ""))
                tts.append(tit + " " + src)

            tot += 1
            if tts and exp in tts[0]:
                h1 += 1
            if any(exp in t for t in tts):
                hk += 1

        print("всего", tot)
        print("попадание 1", h1 / tot)
        print(f"попадание {self.topk}", hk / tot)
