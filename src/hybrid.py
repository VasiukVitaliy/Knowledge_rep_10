#!/usr/bin/env python3
"""ЛР 10, Результат 4: гібридний пошук (BM25 + dense, RRF) і reranking.

Що робить:
  1. BM25 (rank_bm25) і dense-пошук (FAISS HNSW з index.py, косинус) над чанками;
  2. злиття RRF (k = 60);
  3. reranker (cross-encoder): топ-30 після RRF → топ-5;
  4. pre-filter за метаданими (type, sensitivity, date) — кандидати відсікаються ДО ранжування;
  5. оцінює 4 конфігурації (bm25, dense, hybrid, hybrid+rerank) на eval/index_queries.jsonl,
     автоматично знаходить 3 запити, де гібрид виграє, і 1, де програє;
  6. пише docs/hybrid_report.md.

Залежності: pip install numpy sentence-transformers faiss-cpu rank_bm25
Запуск:     python src/hybrid.py --strategy structure
            python src/hybrid.py --model hash --reranker overlap      (швидка перевірка без моделей)
Демо наживо (для захисту):
            python src/hybrid.py --ask "що таке зелена хвиля" --type новина --date-from 2026-01-01
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from chunking import embed, load_docs, load_model, load_queries, norm
except ImportError:  # файл раніше називався chuncking.py
    from chuncking import embed, load_docs, load_model, load_queries, norm
from index import HnswStore, load_chunks

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise SystemExit("Потрібен rank_bm25: pip install rank_bm25")

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
INF = 999

# Фільтри для перевірки pre-filter (назва, умова). Змінюйте під свій корпус.
FILTERS = [
    ("type = новина", {"type": ["новина"]}),
    ("type = статистика", {"type": ["статистика"]}),
    ("type = нормативний", {"type": ["нормативний"]}),
    ("date ≥ 2026-03-01", {"date_from": "2026-03-01"}),
    ("sensitivity = public, type = новина, date ≥ 2026-01-01",
     {"sensitivity": ["public"], "type": ["новина"], "date_from": "2026-01-01"}),
]


# ----------------------------------------------------------------------------
# Токенізація та фільтри
# ----------------------------------------------------------------------------
def tokenize(text, stem=6):
    """Слова нижнім регістром; грубий «стемінг» для української — перші `stem` літер
    (усуває більшість закінчень: світлофор/світлофора/світлофорів)."""
    toks = [t for t in re.findall(r"\w+", text.casefold()) if len(t) > 1]
    return [t[:stem] for t in toks] if stem else toks


def passes(meta, flt):
    """meta — словник із type/sensitivity/date; flt — умова фільтра (або None)."""
    if not flt:
        return True
    if flt.get("type") and meta["type"] not in flt["type"]:
        return False
    if flt.get("sensitivity") and meta["sensitivity"] not in flt["sensitivity"]:
        return False
    if flt.get("date_from") and meta["date"] < flt["date_from"]:
        return False
    if flt.get("date_to") and meta["date"] > flt["date_to"]:
        return False
    return True


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d))."""
    sc = defaultdict(float)
    for lst in rank_lists:
        for r, i in enumerate(lst, 1):
            sc[i] += 1.0 / (k + r)
    return sorted(sc, key=lambda i: (-sc[i], i))


# ----------------------------------------------------------------------------
# Rerankers
# ----------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(self, name):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(name)

    def score(self, query, texts):
        return self.model.predict([(query, t) for t in texts], show_progress_bar=False)


class OverlapReranker:
    """Заглушка без моделі: частка лексем запиту в тексті (лише для перевірки пайплайну)."""

    def score(self, query, texts):
        qt = set(tokenize(query))
        return [len(qt & set(tokenize(t))) / (len(qt) or 1) for t in texts]


def load_reranker(name):
    if name == "none":
        return None
    if name == "overlap":
        return OverlapReranker()
    return CrossEncoderReranker(name)


# ----------------------------------------------------------------------------
# Пошуковець
# ----------------------------------------------------------------------------
class Retriever:
    def __init__(self, chunks, model, e5, reranker, args):
        self.chunks, self.model, self.e5, self.reranker = chunks, model, e5, reranker
        self.pool, self.rrf_k, self.rerank_in = args.pool, args.rrf_k, args.rerank_in
        self.stem = args.stem
        vecs = embed(model, [c["text"] for c in chunks], "passage: " if e5 else "")
        self.store = HnswStore(vecs.shape[1], args.m, args.ef_construction, args.ef_search)
        self.store.add(chunks, vecs)  # внутрішній id збігається з індексом у chunks
        self.bm25 = BM25Okapi([tokenize(c["text"], self.stem) for c in chunks])
        self.types = np.array([c["type"] for c in chunks])
        self.sens = np.array([c["sensitivity"] for c in chunks])
        self.dates = np.array([c["date"] for c in chunks])

    def mask(self, flt):
        """Булева маска чанків, що проходять pre-filter (None — без фільтра)."""
        if not flt:
            return None
        m = np.ones(len(self.chunks), dtype=bool)
        if flt.get("type"):
            m &= np.isin(self.types, flt["type"])
        if flt.get("sensitivity"):
            m &= np.isin(self.sens, flt["sensitivity"])
        if flt.get("date_from"):
            m &= self.dates >= flt["date_from"]
        if flt.get("date_to"):
            m &= self.dates <= flt["date_to"]
        return m

    def bm25_rank(self, q, mask, n):
        scores = self.bm25.get_scores(tokenize(q, self.stem))
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:n]
        return [int(i) for i in order if scores[i] > 0]  # лише з ненульовим збігом

    def dense_rank(self, qv, mask, n):
        if mask is None:
            return self.store.search(qv, n)  # наближений HNSW
        sims = np.where(mask, self.store.vecs @ qv, -np.inf)  # pre-filter: точний пошук по підмножині
        order = np.argsort(-sims, kind="stable")[:n]
        return [int(i) for i in order if np.isfinite(sims[i])]

    def search(self, q, mode, flt, n):
        """Повертає впорядкований список внутрішніх id (до n штук)."""
        mask = self.mask(flt)
        if mode == "bm25":
            return self.bm25_rank(q, mask, n)
        qv = embed(self.model, [q], "query: " if self.e5 else "")[0]
        if mode == "dense":
            return self.dense_rank(qv, mask, n)
        fused = rrf([self.bm25_rank(q, mask, self.pool), self.dense_rank(qv, mask, self.pool)], self.rrf_k)
        if mode == "hybrid":
            return fused[:n]
        cand = fused[:self.rerank_in]  # топ-30 → reranker → топ-n
        if not cand:
            return []
        sc = np.asarray(self.reranker.score(q, [self.chunks[i]["text"] for i in cand]))
        return [cand[i] for i in np.argsort(-sc, kind="stable")][:n]


# ----------------------------------------------------------------------------
# Оцінювання
# ----------------------------------------------------------------------------
def run_mode(R, mode, queries, k, flt=None):
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        ids = R.search(q["q"], mode, flt, k * 6)  # запас на дедуплікацію ієрархічних дітей
        ms = (time.perf_counter() - t0) * 1000
        ctxs, seen = [], set()
        for i in ids:
            c = R.chunks[i]
            key = c.get("parent_id") or c["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            ctxs.append(c)
            if len(ctxs) == k:
                break
        rank = next((r for r, c in enumerate(ctxs, 1)
                     if any(norm(e) in norm(c["context"]) for e in q["evidence"])), None)
        rows.append({"rank": rank, "ms": ms, "top": ctxs,
                     "keys": [c.get("parent_id") or c["chunk_id"] for c in ctxs]})
    return rows


def summarize(rows):
    ms = np.array([r["ms"] for r in rows])
    return {"recall": float(np.mean([r["rank"] is not None for r in rows])),
            "mrr": float(np.mean([1 / r["rank"] if r["rank"] else 0 for r in rows])),
            "ms": float(ms.mean()), "p95": float(np.percentile(ms, 95))}


def pick_examples(res, queries):
    """3 запити, де гібрид кращий за гірший з двох методів (і не гірший за кращий), і 1, де програє."""
    r = lambda x: x if x is not None else INF
    wins, losses = [], []
    for qi in range(len(queries)):
        b, d, h = (r(res[m][qi]["rank"]) for m in ("bm25", "dense", "hybrid"))
        if h < INF and h <= min(b, d) and h < max(b, d):
            wins.append((max(b, d) - h, h < min(b, d), qi))
        if h > min(b, d):
            losses.append((h - min(b, d), qi))
    wins.sort(key=lambda x: (-x[1], -x[0], x[2]))
    losses.sort(key=lambda x: (-x[0], x[1]))
    return [w[2] for w in wins[:3]], [l[1] for l in losses[:1]]


def hint(res, qi, win):
    b, d, h = (res[m][qi]["rank"] for m in ("bm25", "dense", "hybrid"))
    if win:
        if b is None or d is None:
            who = "BM25" if b is None else "dense"
            return (f"{who} не знайшов відповіді в топ-5, інший метод знайшов; RRF підняв чанк "
                    "зі списку того методу, що спрацював.")
        return ("Обидва методи мають відповідь, але не на першому місці; чанк у списках обох, "
                "тому сума 1/(60+rank) обійшла чанки, що сильні лише в одному списку.")
    return ("Один метод ставив відповідь вище, але в злитті її обігнали чанки, що мають середні "
            "позиції в обох списках; RRF не враховує «впевненість» окремого методу.")


# ----------------------------------------------------------------------------
# Звіт
# ----------------------------------------------------------------------------
def snippet(c, n=170):
    s = re.sub(r"\s+", " ", c["context"]).strip()
    return s[:n] + ("…" if len(s) > n else "")


def write_report(root, args, n_chunks, queries, stats, res, rerank_changes, filt_rows,
                 wins, losses, bad, versions):
    L = [
        "# Гібридний пошук і reranking", "",
        f"Стратегія chunking: `{args.strategy}`; чанків: {n_chunks}; запитів: {len(queries)}. "
        f"Модель embeddings: `{args.model}`; reranker: `{args.reranker}`. Версії: {versions}.", "",
        "Таблиця 1 — Параметри", "", "| Параметр | Значення |", "|---|---|",
        f"| BM25 | rank_bm25 (BM25Okapi), лексеми = перші {args.stem} літер слова |",
        f"| Dense | FAISS HNSW, M={args.m}, efConstruction={args.ef_construction}, efSearch={args.ef_search} |",
        f"| Злиття | RRF, k = {args.rrf_k}; у кожному списку по {args.pool} кандидатів |",
        f"| Reranking | топ-{args.rerank_in} після RRF → топ-{args.k} |",
        f"| Pre-filter | type, sensitivity, date (відсікає кандидатів до ранжування) |", "",
        "Таблиця 2 — Порівняння конфігурацій", "",
        f"| Конфігурація | recall@{args.k} | MRR@{args.k} | Затримка, мс (сер. / p95) |", "|---|---|---|---|"]
    for m in MODES:
        if m in stats:
            s = stats[m]
            L.append(f"| {m} | {s['recall']:.2f} | {s['mrr']:.2f} | {s['ms']:.1f} / {s['p95']:.1f} |")
    if rerank_changes is not None:
        L += ["", f"Reranker змінив топ-{args.k} відносно hybrid у **{rerank_changes['set']}** із "
                  f"{len(queries)} запитів (склад) і у **{rerank_changes['order']}** (порядок)."]
    present = [m for m in MODES if m in res]
    L += ["", f"Таблиця 3 — Позиція першого влучання в топ-{args.k} («—» якщо немає)", "",
          "| № | Запит | " + " | ".join(present) + " |", "|---|---|" + "---|" * len(present)]
    for qi, q in enumerate(queries):
        L.append(f"| {qi + 1} | {q['q']} | " + " | ".join(str(res[m][qi]["rank"] or "—") for m in present) + " |")
    if bad:
        L += ["", "Запити, виключені з оцінки (evidence не знайдено в корпусі):", ""]
        L += [f"- рядок {ln}: «{q}» — {m}" for ln, q, m in bad]

    L += ["", "## Pre-filter за метаданими", "",
          "Фільтр застосовується до ранжування: BM25 обнуляє, а dense виключає чанки, що не проходять умову. "
          "«Вибірка» — запити, для яких еталонна фраза є хоча б в одному документі, що проходить фільтр. "
          "«Витоки» — скільки чанків у видачі порушують умову (має бути 0); "
          "«Влучання поза вибіркою» — скільки запитів знайшли відповідь у документі, що фільтр відсікає (має бути 0).", "",
          "| Фільтр | Чанків проходить | Запитів у вибірці | recall@5 на вибірці | Витоки | Влучання поза вибіркою |",
          "|---|---|---|---|---|---|"]
    for r in filt_rows:
        rec = "—" if r["n_elig"] == 0 else f"{r['recall']:.2f}"
        L.append(f"| {r['name']} | {r['n_pass']} | {r['n_elig']} | {rec} | {r['leaks']} | {r['outside_hits']} |")

    L += ["", "## Приклади: де гібрид виграє і де програє", ""]
    def block(qi, title, win):
        q = queries[qi]
        out = [f"### {title}", "", f"**Запит {qi + 1}:** {q['q']}", "",
               f"Еталонна фраза: «{q['evidence'][0]}»", "",
               f"Позиції: BM25 — {res['bm25'][qi]['rank'] or '—'}, dense — {res['dense'][qi]['rank'] or '—'}, "
               f"hybrid — {res['hybrid'][qi]['rank'] or '—'}"
               + (f", hybrid+rerank — {res['hybrid_rerank'][qi]['rank'] or '—'}" if "hybrid_rerank" in res else ""), "",
               "Топ-1 кожного методу:", ""]
        for m in present:
            top = res[m][qi]["top"]
            out.append(f"- **{m}:** " + (snippet(top[0]) if top else "(порожньо)"))
        out += ["", f"Чому (автопідказка): {hint(res, qi, win)}", "",
                "> TODO: поясніть власними словами на цьому прикладі.", ""]
        return out
    for n, qi in enumerate(wins, 1):
        L += block(qi, f"Виграш {n}", True)
    if len(wins) < 3:
        L += [f"> Знайдено лише {len(wins)} з 3 прикладів виграшу: додайте запити, де слова запиту не збігаються з "
              "текстом (перефразування) або навпаки містять точні назви/числа.", ""]
    for qi in losses:
        L += block(qi, "Програш", False)
    if not losses:
        L += ["> Автоматично програшу не знайдено (hybrid не гірший за кращий із BM25/dense на всіх запитах). "
              "Додайте складніші запити та повторіть запуск.", ""]
    L += ["## Висновок", "",
          "> TODO: власними словами — коли гібрид і reranker варті додаткової затримки, спираючись на таблиці 2–3.", ""]
    path = root / "docs/hybrid_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
def ask(R, text, args, flt):
    """Демо одного запиту: топ-k для кожної конфігурації."""
    for m in MODES:
        if m == "hybrid_rerank" and R.reranker is None:
            continue
        ids = R.search(text, m, flt, args.k)
        print(f"\n== {m} ==")
        for r, i in enumerate(ids, 1):
            c = R.chunks[i]
            print(f"{r}. [{c['type']}, {c['date']}, {c['sensitivity']}] {c['doc_id']} — {snippet(c, 120)}")
        if not ids:
            print("(нічого не знайдено)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strategy", default="structure", choices=["fixed", "structure", "hierarchical"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-small", help="модель embeddings або 'hash'")
    ap.add_argument("--reranker", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                    help="cross-encoder, 'overlap' (заглушка) або 'none'")
    ap.add_argument("--queries", default="eval/index_queries.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--pool", type=int, default=50, help="кандидатів від кожного методу перед RRF")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rerank-in", type=int, default=30)
    ap.add_argument("--stem", type=int, default=6, help="довжина «стему» для BM25 (0 — вимкнути)")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--ask", help="виконати один запит і вивести топ-k для кожної конфігурації")
    ap.add_argument("--type", nargs="*", help="pre-filter: тип документа")
    ap.add_argument("--sensitivity", nargs="*", help="pre-filter: sensitivity")
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    args = ap.parse_args()
    root = Path(args.root)

    chunks = load_chunks(root, args.strategy)
    e5 = "e5" in args.model.lower()
    reranker = load_reranker(args.reranker)
    R = Retriever(chunks, load_model(args.model), e5, reranker, args)
    print(f"Чанків ({args.strategy}): {len(chunks)}")

    if args.ask:
        flt = {k: v for k, v in {"type": args.type, "sensitivity": args.sensitivity,
                                 "date_from": args.date_from, "date_to": args.date_to}.items() if v}
        ask(R, args.ask, args, flt or None)
        return

    docs = load_docs(root)
    corpus_norm = norm(" ".join(t for _, _, t in docs))
    qpath = root / args.queries
    if not qpath.exists():
        raise SystemExit(f"Немає {qpath}")
    queries, bad = load_queries(qpath, corpus_norm)
    for ln, q, m in bad:
        print(f"[warn] запит у рядку {ln} виключено: evidence не знайдено в корпусі: {m}")
    if not queries:
        raise SystemExit("Немає придатних запитів")

    modes = [m for m in MODES if m != "hybrid_rerank" or reranker is not None]
    for m in modes:  # прогрів (завантаження ваг, кеші)
        R.search(queries[0]["q"], m, None, args.k)
    res, stats = {}, {}
    for m in modes:
        res[m] = run_mode(R, m, queries, args.k)
        stats[m] = summarize(res[m])
        s = stats[m]
        print(f"{m:14s} recall@{args.k}={s['recall']:.2f} MRR={s['mrr']:.2f} {s['ms']:.1f} мс")

    rerank_changes = None
    if "hybrid_rerank" in res:
        rerank_changes = {
            "set": sum(set(res["hybrid"][i]["keys"]) != set(res["hybrid_rerank"][i]["keys"]) for i in range(len(queries))),
            "order": sum(res["hybrid"][i]["keys"] != res["hybrid_rerank"][i]["keys"] for i in range(len(queries))),
        }
        print(f"reranker змінив топ-{args.k}: склад у {rerank_changes['set']}, порядок у {rerank_changes['order']} запитах")

    # --- перевірка pre-filter ---
    doc_meta = {}
    for c in chunks:
        doc_meta.setdefault(c["doc_id"], {k: c[k] for k in ("type", "sensitivity", "date")})
    evidence_docs = [{d for d, _, t in docs if any(norm(e) in norm(t) for e in q["evidence"])} for q in queries]
    filt_rows = []
    for name, flt in FILTERS:
        n_pass = sum(passes(c, flt) for c in chunks)
        elig = [qi for qi in range(len(queries))
                if any(passes(doc_meta[d], flt) for d in evidence_docs[qi] if d in doc_meta)]
        mode = "hybrid_rerank" if "hybrid_rerank" in res else "hybrid"
        rows = run_mode(R, mode, queries, args.k, flt) if n_pass else [{"rank": None, "top": []}] * len(queries)
        leaks = sum(not passes(c, flt) for r in rows for c in r["top"])
        outside = sum(rows[qi]["rank"] is not None for qi in range(len(queries)) if qi not in set(elig))
        recall = float(np.mean([rows[qi]["rank"] is not None for qi in elig])) if elig else 0.0
        filt_rows.append({"name": name, "n_pass": n_pass, "n_elig": len(elig), "recall": recall,
                          "leaks": leaks, "outside_hits": outside})
        print(f"[filter] {name}: чанків {n_pass}, витоків {leaks}, влучань поза вибіркою {outside}")

    wins, losses = pick_examples(res, queries)
    try:
        import sentence_transformers
        import faiss
        versions = (f"sentence-transformers {sentence_transformers.__version__}, faiss {faiss.__version__}, "
                    f"numpy {np.__version__}")
    except ImportError:
        versions = f"numpy {np.__version__}"
    path = write_report(root, args, len(chunks), queries, stats, res, rerank_changes, filt_rows,
                        wins, losses, bad, versions)
    print(f"\nЗвіт: {path}")


if __name__ == "__main__":
    main()