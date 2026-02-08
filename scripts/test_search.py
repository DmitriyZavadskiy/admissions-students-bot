from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLL = "admissions_chunks"
EMBMDL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

QS = [
    "стоимость обучения на прикладную математику и информатику",
    "какие документы нужны для поступления",
    "сроки подачи документов",
    "общежитие первокурсникам как получить",
    "контакты приемной комиссии",
]

class SmokeSearch:
    def __init__(self, coll: str = COLL, mdl: str = EMBMDL, url: str = "http://localhost:6333"):
        self.coll = coll
        self.mdl = mdl
        self.url = url

    def _qry(self, qdr: QdrantClient, emb: SentenceTransformer, que: str, lim: int = 5):
        qvec = emb.encode(que, normalize_embeddings=True).tolist()

        res = qdr.query_points(
            collection_name=self.coll,
            query=qvec,
            limit=lim,
            with_payload=True,
        )

        return res.points if hasattr(res, "points") else res

    def run(self, qs: list[str] = QS, lim: int = 5) -> None:
        qdr = QdrantClient(url=self.url)
        emb = SentenceTransformer(self.mdl)

        for que in qs:
            hits = self._qry(qdr, emb, que, lim=lim)

            print()
            print("Вопрос:", que)
            for idx, hit in enumerate(hits, 1):
                pay = getattr(hit, "payload", None) or {}
                scr = getattr(hit, "score", None)
                tit = pay.get("title", "")
                src = pay.get("source", "")
                print(idx, "оценка", round(float(scr), 4) if scr is not None else scr, "|", tit, "|", src)
