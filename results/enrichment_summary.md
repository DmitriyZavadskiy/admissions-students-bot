# Доработки под фидбек руководителя — сводка результатов

Фазы 1-3 вкачены в проект (скрипты в `scripts/`, артефакт `data/processed/enriched_chunks.json`,
`index_text` в `rag.py`/`indexes_for_Qadr.py`). Этот документ — сводка чисел для Фазы 4 (LaTeX).

Фидбек, который покрываем:
1. В таблицах экспериментов жирным выделять победившую конфигурацию. *(правка LaTeX, делается при редактуре отчёта — данные для жирного ниже)*
2. В реранке замерить **latency**. ✅
3. Для качества ретрива по теме **cost** — обогащение метаданными чанков
   (опорные вопросы, суммаризация, соседние чанки). ✅
4. Дописать Заключение и Further Work. *(текст LaTeX, на основе результатов ниже)*

Дополнительно по нашему обсуждению:
- Согласованность выборки: всё пересчитано на **N=65** (раньше 5.5/5.6/5.8 были на 50). ✅

---

## Окружение (важно!)
Чтобы считать на GPU (RTX 3070), переустановлена **CUDA-сборка** `llama-cpp-python`
(0.3.22 CPU → 0.3.23 cu124, `supports_gpu_offload=True`). Генерация Qwen идёт на GPU
(~45 tok/s, ~1.1 c/ответ). `torch` остался CPU-сборкой (cu130 vs драйвер 12.8) —
эмбеддинги/реранкер считаются на CPU, но они быстрые, узкого места нет.
Латентность реранкера ниже — это **CPU**-числа (что реалистично: по design-doc
retrieval/rerank допускают CPU).

---

## 1. Согласованность N=65  (`results/n65/`)
Пересчёт baseline (5.5), чанкинга (5.6) и retrieval (5.8) на `gold_qa_extended.json`
(65 вопросов). **Выводы отчёта не изменились.**

| Раздел | Победитель (N=65) | Hit@1 | Hit@5 | MRR | MAP | NDCG@5 |
|---|---|---|---|---|---|---|
| 5.5 baseline (MiniLM dense) | — | 0.138 | 0.523 | 0.313 | 0.318 | 0.282 |
| 5.6 чанкинг | **fixed 700/100** | 0.200 | 0.523 | 0.339 | 0.321 | 0.319 |
| 5.8 retrieval | **hybrid_e5** | 0.538 | 0.846 | 0.667 | 0.609 | 0.598 |

Полные таблицы — в `results/n65/*/summary.csv`.

## 2. Latency реранкера  (`results/reranker_latency/latency.csv`)
Время одного `reranker.predict` на запрос (65 вопросов, CPU, прогрев исключён):

| top_n | mmarco mean / p95 (мс) | msmarco mean / p95 (мс) |
|---|---|---|
| 10 | 418 / 481 | 645 / 675 |
| 15 | **638 / 733** | 940 / 981 |
| 25 | 1162 / 1415 | 1609 / 1648 |
| 50 | 2432 / 3072 | 3206 / 3426 |

Вывод: latency растёт ~линейно по `top_n`; `mmarco` дешевле и качественнее `msmarco`.
`mmarco_t15` — лучший компромисс качество/задержка; `t50` почти не улучшает метрики,
но в ~4× дороже. Усиливает тезис «реранкер нельзя включать вслепую».

## 3. Обогащение чанков для cost  (`data/enriched_chunks.json`, `results/retrieval_enriched/`)
Что добавлено к чанкам (`scripts/enrich_chunks.py`):
- **price_row** — 63 структурных мини-чанка «программа → цена» из таблицы
  стоимости (Приложение 1, Москва). Покрывают **25/25** cost-программ.
- **aux_questions** — 3-4 опорных вопроса на чанк (Qwen, GPU).
- **summary** — 1 предложение о чём фрагмент (Qwen, GPU).
- **neighbor_ids** — id соседних чанков документа.
- **index_text** = summary + aux_questions + text (по нему идёт индексация).

### Retrieval (hybrid e5+BM25, N=65, по категориям)
| | baseline | struct (цены) | full (+LLM) |
|---|---|---|---|
| Overall Hit@1 | 0.554 | **0.800** | 0.754 |
| Overall Hit@5 | 0.862 | 0.938 | **0.985** |
| Overall MRR | 0.687 | **0.854** | 0.851 |
| Overall NDCG@5 | 0.614 | 0.658 | **0.709** |
| **cost** Hit@1 | 0.320 | **1.000** | 0.920 |
| **cost** MRR | 0.546 | **1.000** | 0.950 |
| contacts Hit@5 | 0.333 | 0.667 | **1.000** |
| documents Hit@1 | 0.333 | 0.333 | **0.667** |

Вывод: **структурные мини-чанки чинят точность на cost (Hit@1 0.32→1.0), а
LLM-обогащение (опорные вопросы) поднимает recall на длинном хвосте (Hit@5 0.86→0.985,
NDCG 0.61→0.71)**. Другие категории не деградируют.

### Генерация (Qwen, N=65)  (`results/gen_enriched/`)
| acc | baseline (`gen_v2_cost`) | enriched + соседи | **enriched без соседей** |
|---|---|---|---|
| Overall | 0.554 | 0.600 | **0.723** |
| **cost** | 0.360 | 0.760 | **0.840** |
| dates | 0.600 | 0.400 | 0.600 |
| documents | 0.333 | 0.333 | 0.667 |
| hallucination_free | 1.000 | 0.985 | 0.985 |
| source_reference | 1.000 | 1.000 | 1.000 |

Вывод: обогащённый индекс поднимает генерацию **0.55→0.72 overall и cost 0.36→0.84**.
**Соседние чанки в промпте генерации не помогают** — размывают ответ и роняют dates
(0.60→0.40). Рекомендация: соседей в контекст генерации не добавлять (или только
выборочно), что согласуется с категориально-адаптивной логикой проекта (как с реранкером).

---

## Файлы
```
scripts/enrich_chunks.py      # обогащение чанков (соседи, таблица цен, aux/summary, index_text)
scripts/eval_enriched.py      # retrieval baseline vs enriched по категориям
scripts/gen_enriched.py       # генерация Qwen с/без соседних чанков
scripts/reranker_latency.py   # latency реранкеров по top_n
data/enriched_chunks.json     # обогащённый набор (448 чанков: 385 + 63 price_row)
data/enriched_chunks_nollm.json # быстрый вариант без LLM (только структура+соседи)
results/n65/                  # пересчёт 5.5/5.6/5.8 на N=65
results/retrieval_enriched/   # baseline / enriched_struct / enriched_full + by_category
results/gen_enriched/         # генерация с/без соседей + by_category
results/reranker_latency/     # latency.csv / latency.json
```

## Воспроизвести (из корня репозитория)
Скрипты запускаются через диспетчер `python -m scripts <cmd>` (как `python -m scripts <cmd>` в репозитории).
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts enrich_chunks --out data/processed/enriched_chunks.json
python -m scripts eval_enriched --chunks data/processed/preprocessed_chunks/prep_heads.json --field text --label baseline
python -m scripts eval_enriched --chunks data/processed/enriched_chunks.json --field index_text --label enriched_full
python -m scripts gen_enriched --chunks data/processed/enriched_chunks.json --field index_text --label gen_enriched_noneigh --no-neighbors
python -m scripts reranker_latency
# N=65
python -m scripts eval_retrieval --eval data/eval/gold_qa_extended.json --label baseline --results-dir results/n65/baseline_65
python -m scripts chunk_experiments --eval data/eval/gold_qa_extended.json --results-dir results/n65/chunks_65
python -m scripts retrieval_experiments --eval data/eval/gold_qa_extended.json --rerank --results-dir results/n65/retrieval_65
```

## Как вкатить (когда скажешь)
- `scripts/enrich_chunks.py`, `eval_enriched.py`, `gen_enriched.py`,
  `reranker_latency.py` → перенести в `scripts/`, импорты `new_changes.scripts.X`
  → `scripts.X`, зарегистрировать команды в `scripts/__main__.py`.
- В `scripts/retrieval_experiments.py` и `rag.py` — индексировать `index_text`
  (`chunk.get("index_text") or chunk["text"]`), если используем обогащённый набор.
- Результаты `results/*` → в `results/` (или оставить как есть).
- LaTeX: проставить жирным победителей (данные выше), заменить таблицы на N=65,
  добавить подраздел про обогащение+latency, дописать Заключение и Further Work.
```
