import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

COLL = "admissions_chunks"
EMBMDL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHFP = Path("data/processed/chunks.json")

BCHSZ = 256


class QdrantIndexer:
    def __init__(self, coll: str = COLL, mdl: str = EMBMDL, chfp: Path = CHFP, bsz: int = BCHSZ):
        self.coll = coll
        self.mdl = mdl
        self.chfp = chfp
        self.bsz = bsz

    @staticmethod
    def rd(pth: Path):
        with pth.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def run(self) -> int:
        qdr = QdrantClient(host="localhost", port=6333)
        emb = SentenceTransformer(self.mdl)

        vdim = emb.get_sentence_embedding_dimension()

        try:
            qdr.create_collection(
                collection_name=self.coll,
                vectors_config=models.VectorParams(size=vdim, distance=models.Distance.COSINE),
            )
        except Exception:
            pass

        bch: list[models.PointStruct] = []
        cnt = 0

        for rec in self.rd(self.chfp):
            txt = rec["text"]
            vct = emb.encode(txt, normalize_embeddings=True).tolist()

            pay = {
                "doc_id": rec["doc_id"],
                "source": rec["source"],
                "title": rec["title"],
                "type": rec.get("type"),
                "chunk_id": int(rec["chunk_id"]),
                "start_char": int(rec.get("start_char", 0)),
                "end_char": int(rec.get("end_char", 0)),
                "text": txt,
            }

            bch.append(models.PointStruct(id=int(rec["chunk_id"]), vector=vct, payload=pay))
            cnt += 1

            if len(bch) >= self.bsz:
                qdr.upsert(collection_name=self.coll, points=bch)
                bch = []

        if bch:
            qdr.upsert(collection_name=self.coll, points=bch)

        print(f"проиндексировано {cnt} чанков в {self.coll}")
        return cnt
