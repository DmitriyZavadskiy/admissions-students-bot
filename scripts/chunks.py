import json
import re
from pathlib import Path

SRC = Path("data/processed/documents.json")
DST = Path("data/processed/chunks.json")

MAXC = 1600
OVLP = 250

SRE = re.compile(r"(?<=[\.\!\?\:])\s+|\n+")


class ChunkMaker:
    def __init__(self, src: Path = SRC, dst: Path = DST, maxc: int = MAXC, ovlp: int = OVLP):
        self.src = src
        self.dst = dst
        self.maxc = maxc
        self.ovlp = ovlp

    @staticmethod
    def rd(pth: Path):
        with pth.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def run(self) -> int:
        self.dst.parent.mkdir(parents=True, exist_ok=True)

        cid = 0
        arr = []
        for doc in self.rd(self.src):
            txt = doc["text"]
            prt = [p.strip() for p in SRE.split(txt) if p.strip()]

            buf = ""
            stch = 0
            for seg in prt:
                if not buf:
                    buf = seg
                    continue

                if len(buf) + 1 + len(seg) <= self.maxc:
                    buf += "\n" + seg
                else:
                    ench = stch + len(buf)
                    rec = {
                        "chunk_id": cid,
                        "doc_id": doc["id"],
                        "source": doc["source"],
                        "title": doc["title"],
                        "type": doc.get("type", "unknown"),
                        "text": buf,
                        "start_char": stch,
                        "end_char": ench,
                    }
                    arr.append(rec)
                    cid += 1

                    ovlp = buf[-self.ovlp :] if len(buf) > self.ovlp else buf
                    stch = max(0, ench - len(ovlp))
                    buf = ovlp + "\n" + seg

            if buf.strip():
                ench = stch + len(buf)
                rec = {
                    "chunk_id": cid,
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "type": doc.get("type", "unknown"),
                    "text": buf,
                    "start_char": stch,
                    "end_char": ench,
                }
                arr.append(rec)
                cid += 1

        with self.dst.open("w", encoding="utf-8") as out:
            json.dump(arr, out, ensure_ascii=False)
            out.write("\n")

        print(f"сохранено {cid} чанков в {self.dst}")
        return cid
