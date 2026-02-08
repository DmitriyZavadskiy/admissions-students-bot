import json
import re
from collections import Counter as Cnt
from pathlib import Path

import fitz
import bs4
from requests import get
import trafilatura as traf

PDFD = Path("data/raw/pdfs")
UTXT = Path("data/raw/urls.txt")
OUTP = Path("data/processed/documents.json")


class DocsParser:
    def __init__(self, pdfd: Path = PDFD, urlp: Path = UTXT, outp: Path = OUTP):
        self.pdfd = pdfd
        self.urlp = urlp
        self.outp = outp

    @staticmethod
    def norm_space(txt: str) -> str:
        txt = txt.replace("\u00a0", " ")
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    @staticmethod
    def drop_hf(pgs: list[str]) -> list[str]:
        if len(pgs) < 3:
            return pgs

        top = Cnt()
        bot = Cnt()
        spl: list[list[str]] = []

        for txt in pgs:
            lns = [x.strip() for x in txt.splitlines() if x.strip()]
            spl.append(lns)
            if not lns:
                continue
            for x in lns[:2]:
                top[x] += 1
            for x in lns[-2:]:
                bot[x] += 1

        thr = max(2, int(0.7 * len(pgs)))

        topb = {k for k, v in top.items() if v >= thr and len(k) <= 80}
        botb = {k for k, v in bot.items() if v >= thr and len(k) <= 80}

        out: list[str] = []
        for lns in spl:
            if lns and lns[0] in topb:
                lns = lns[1:]
            if lns and lns[0] in topb:
                lns = lns[1:]
            if lns and lns[-1] in botb:
                lns = lns[:-1]
            if lns and lns[-1] in botb:
                lns = lns[:-1]
            out.append("\n".join(lns))

        return out

    def parse_pdf(self, pth: Path) -> str:
        with fitz.open(pth) as doc:
            pgs: list[str] = []

            for idx in range(len(doc)):
                pag = doc.load_page(idx)

                blks = pag.get_text("blocks")
                blks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
                prts: list[str] = []
                for blk in blks:
                    btx = blk[4]
                    if not btx:
                        continue
                    prts.append(btx)
                ptxt = "\n".join(prts)
                ptxt = self.norm_space(ptxt)
                pgs.append(ptxt)

        pgs = self.drop_hf(pgs)
        txt = "\n\n".join([p for p in pgs if p.strip()])
        return self.norm_space(txt)

    def parse_html(self, url: str) -> tuple[str, str]:
        rsp = get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
        rsp.raise_for_status()

        html = rsp.text
        tit = url

        if traf is not None:
            txt = traf.extract(html, include_comments=False, include_tables=True)
            if txt and txt.strip():
                sup = bs4.BeautifulSoup(html, "html.parser")
                if sup.title:
                    tit = sup.title.get_text(strip=True)
                return tit, self.norm_space(txt)

        sup = bs4.BeautifulSoup(html, "html.parser")
        if sup.title:
            tit = sup.title.get_text(strip=True)

        for tag in sup(["script", "style", "noscript"]):
            tag.decompose()

        txt = sup.get_text("\n", strip=True)
        return tit, self.norm_space(txt)

    def load_urls(self) -> list[str]:
        if not self.urlp.exists():
            return []
        txt = self.urlp.read_text(encoding="utf-8")
        uls = [u.strip() for u in txt.splitlines() if u.strip()]
        return uls

    def run(self) -> int:
        self.outp.parent.mkdir(parents=True, exist_ok=True)

        uls = self.load_urls()

        did = 0
        fst = True
        with self.outp.open("w", encoding="utf-8") as fp:
            fp.write("[\n")
            for pdf in sorted(self.pdfd.glob("*.pdf")):
                txt = self.parse_pdf(pdf)
                rec = {
                    "id": f"pdf_{did}",
                    "source": str(pdf),
                    "title": pdf.name,
                    "type": "pdf",
                    "text": txt,
                }
                if not fst:
                    fp.write(",\n")
                fp.write(json.dumps(rec, ensure_ascii=False))
                fst = False
                did += 1

            for url in uls:
                tit, txt = self.parse_html(url)
                rec = {
                    "id": f"url_{did}",
                    "source": url,
                    "title": tit,
                    "type": "html",
                    "text": txt,
                }
                if not fst:
                    fp.write(",\n")
                fp.write(json.dumps(rec, ensure_ascii=False))
                fst = False
                did += 1
            fp.write("\n]\n")

        print(f"сохранено {did} документов в {self.outp}")
        return did
