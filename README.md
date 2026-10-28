# ЛР 10 — Векторне сховище та семантичний пошук (домен: регулювання дорожнього руху та управління транспортними потоками у Львові)

## Залежності
Python 3.10+; `pip install -r requirements.txt` (sentence-transformers, faiss-cpu, rank_bm25, numpy, requests, beautifulsoup4).
Моделі (завантажуються з Hugging Face): `intfloat/multilingual-e5-small`,
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
Seed усіх випадкових кроків: 42 (`--seed 42`).

## Запуск з нуля (з кореня репозиторію)
| Крок | Команда | Результат |
|---|---|---|
| 1. Корпус | `python src/scrap.py --root . --force` | `data/corpus.jsonl`, `docs/corpus_card.md` (джерела кешуються в `data/raw/`) |
| 2. Chunking | `python src/chunking.py` | `data/chunks/*.jsonl`, `docs/chunking_cmp.md` |
| 3. Індекси | `python src/index.py` | `data/index/*`, `docs/embeddings_cmp.md` |
| 4. Гібрид | `python src/hybrid.py` | `docs/hybrid_report.md` |
| 5. κ-розмітка | `python src/eval_search.py export-kappa --n 12 --seed 42` → двоє розмітників заповнюють `eval/kappa_pool_A.csv` і `kappa_pool_B.csv` → `python src/eval_search.py kappa` |
| 6. Оцінка | `python src/eval_search.py run` | `docs/search_eval.md` |

Чотири необов’язкові джерела (zaxid.net) завантажуються лише за наявності доступу до мережі; якщо вони завантажились, перегляньте нові документи в `data/clean/`, додайте до `eval/gold.jsonl` кілька запитів по них і перевірте, що evidence є в тексті.
Після кроку 1 перегляньте `docs/corpus_card.md` (ліцензії позначені «перевірте»).

## Структура
`src/` — код · `data/` — корпус, чанки, індекси · `eval/` — запити й золотий набір · `docs/` — звіти ·
`llm_log/` — журнал взаємодії з LLM · `claims/claims.csv` — таблиця тверджень · `_old_run/` — результати попереднього прогону (на старому корпусі з 18 документів).

## Конфіденційність
Фрагменти з `sensitivity = restricted` (новини, сторінки без явної ліцензії) не передаються у зовнішні сервіси.
Усі моделі виконуються локально; ключів API у коді немає.
