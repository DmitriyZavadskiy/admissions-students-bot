import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from scripts.abbreviations import expand_query
from scripts.eval_retrieval import read_json
from scripts.generation_eval import GGUF, build_context, generate
from scripts.retrieval_experiments import BM25, RERANKER, minmax, prepare_for_embedder, tokenize


CHFP = Path("data/processed/enriched_chunks.json")
FALLBACK_CHFP = Path("data/processed/chunk_experiments/chunks_fixed_chars_700_100.json")
EMBMDL = "intfloat/multilingual-e5-small"

TOPK = 5
MAXCTX = 1800
MAXTOK = 350
HYBRID_ALPHA = 0.65
RERANK_TOPN = 15

COST_RE = re.compile(
    r"стоим|цен[аиу]|сколько\s+стоит|платн|руб|тыс|обучени[ея]\s+за\s+год",
    re.IGNORECASE,
)


def question_category(que: str) -> str:
    if COST_RE.search(que):
        return "cost"
    return "general"


CJK_RE = re.compile(r"[　-〿㐀-鿿＀-￯]")


def strip_cjk(text: str) -> str:
    text = CJK_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


TAG_RE = re.compile(r"<[^>]{0,60}>")
MAX_QUERY_CHARS = 400


def sanitize_query(text: str) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_QUERY_CHARS]


class RagLocalChat:
    def __init__(
        self,
        chfp: Path = CHFP,
        mdl: str = EMBMDL,
        reranker_mdl: str = RERANKER,
        gguf: Path = GGUF,
        topk: int = TOPK,
        maxctx: int = MAXCTX,
        maxtok: int = MAXTOK,
        alpha: float = HYBRID_ALPHA,
        rerank_topn: int = RERANK_TOPN,
        gpu_layers: int = -1,
        use_llm: bool = True,
    ):
        self.chfp = chfp if chfp.exists() else FALLBACK_CHFP
        self.mdl = mdl
        self.reranker_mdl = reranker_mdl
        self.gguf = gguf
        self.topk = topk
        self.maxctx = maxctx
        self.maxtok = maxtok
        self.alpha = alpha
        self.rerank_topn = rerank_topn
        self.gpu_layers = gpu_layers
        self.use_llm = use_llm

        self.chunks = read_json(self.chfp)
        self.emb = SentenceTransformer(self.mdl)

        texts = [chunk.get("index_text") or chunk["text"] for chunk in self.chunks]
        emb_texts = prepare_for_embedder(self.mdl, texts, "passage")
        self.vectors = self.emb.encode(
            emb_texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        self.bm25 = BM25([tokenize(text) for text in texts])
        self.reranker = CrossEncoder(self.reranker_mdl)
        self.llm = None

        if self.use_llm:
            if not self.gguf.exists():
                raise FileNotFoundError(f"GGUF не найден: {self.gguf}")

            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(self.gguf),
                n_ctx=8192,
                n_threads=max(2, os.cpu_count() or 4),
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )

    def rank(self, que: str) -> tuple[list[int], str]:
        emb_que = self.emb.encode(
            prepare_for_embedder(self.mdl, [que], "query"),
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        dense = np.asarray(self.vectors @ emb_que)
        sparse = self.bm25.score(tokenize(que))
        scores = self.alpha * minmax(dense) + (1 - self.alpha) * minmax(sparse)
        base_rank = list(np.argsort(-scores))

        category = question_category(que)
        if category == "cost":
            return base_rank, category

        candidates = base_rank[: self.rerank_topn]
        pairs = [(que, self.chunks[idx]["text"]) for idx in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        reranked = [
            idx
            for idx, _ in sorted(
                zip(candidates, rerank_scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        return reranked + base_rank[self.rerank_topn :], category

    def answer(self, que: str) -> dict:
        que = expand_query(sanitize_query(que))
        ranking, category = self.rank(que)
        ctx = build_context(self.chunks, ranking, top_k=self.topk, max_chars=self.maxctx)
        ans = ctx
        if self.llm is not None:
            ans = strip_cjk(generate(self.llm, que, ctx, max_tokens=self.maxtok))

        srcs = []
        for idx in ranking[: self.topk]:
            chunk = self.chunks[idx]
            srcs.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                }
            )

        return {
            "category": category,
            "answer": ans,
            "top_sources": srcs,
        }

    def run(self) -> None:
        while True:
            que = input("Вопрос: ").strip()
            if not que:
                continue
            if que.lower() in {"exit", "quit", "выход"}:
                break

            res = self.answer(que)
            print(res["answer"])
            print()
