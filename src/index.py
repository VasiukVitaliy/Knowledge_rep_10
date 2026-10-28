#!/usr/bin/env python3
"""ЛР 10, Результат 3: векторний індекс (FAISS HNSW) і порівняння embeddings.

Що робить:
  1. читає чанки обраної стратегії (data/chunks/<стратегія>.jsonl з chunking.py);
  2. для кожної моделі embeddings будує індекс FAISS HNSW (M, efConstruction, efSearch зафіксовано),
     вимірює час embedding / побудови індексу;
  3. на запитах eval/index_queries.jsonl (потрібно 20) рахує recall@5, MRR@5 і затримку
     (embedding запиту та пошук окремо) для кількох значень efSearch;
  4. перевіряє upsert/delete на 5 документах (м'яке видалення + додавання, компактація);
  5. пише docs/embeddings_cmp.md і зберігає індекси в data/index/.

Формат запитів — той самий, що й у chunk_queries.jsonl:
  {"q": "...", "evidence": ["дослівна фраза з корпусу"]}

Залежності: pip install numpy sentence-transformers faiss-cpu
Запуск:     python src/index.py --strategy structure
            python src/index.py --models hash          (швидка перевірка без моделей)
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from chunking import embed, load_docs, load_model, load_queries, norm
except ImportError:  # файл раніше називався chuncking.py
    from chuncking import embed, load_docs, load_model, load_queries, norm

try:
    import faiss
except ImportError:
    raise SystemExit("Потрібен faiss: pip install faiss-cpu")

DEFAULT_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
REPEAT = 20           # скільки разів повторюємо пошук для усереднення затримки
DEAD_RATIO_LIMIT = 0.3  # при такій частці видалених векторів індекс перебудовується


# ----------------------------------------------------------------------------
# Сховище: HNSW + м'яке видалення
# ----------------------------------------------------------------------------
class HnswStore:
    """FAISS IndexHNSWFlat (косинус через нормовані вектори + inner product).

    HNSW у FAISS не підтримує remove_ids, тому видалення — «надгробки» (alive=False);
    пошук бере з запасом і відфільтровує видалені. Коли видалених забагато — compact().
    """

    def __init__(self, dim, M, ef_construction, ef_search):
        self.dim, self.M, self.ef_construction, self.ef_search = dim, M, ef_construction, ef_search
        self.compactions = 0
        self._reset()

    def _reset(self):
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.vecs = np.zeros((0, self.dim), dtype="float32")
        self.chunks, self.alive = [], []
        self.by_doc = defaultdict(list)

    @property
    def n_alive(self):
        return sum(self.alive)

    @property
    def n_dead(self):
        return len(self.alive) - self.n_alive

    def add(self, chunks, vecs):
        start = len(self.chunks)
        self.index.add(vecs)
        self.vecs = np.vstack([self.vecs, vecs])
        for j, c in enumerate(chunks):
            self.chunks.append(c)
            self.alive.append(True)
            self.by_doc[c["doc_id"]].append(start + j)

    def chunks_of(self, doc_id):
        return [self.chunks[i] for i in self.by_doc.get(doc_id, []) if self.alive[i]]

    def delete_doc(self, doc_id):
        ids = self.by_doc.pop(doc_id, [])
        for i in ids:
            self.alive[i] = False
        if self.alive and self.n_dead / len(self.alive) > DEAD_RATIO_LIMIT:
            self.compact()
        return len(ids)

    def upsert_doc(self, doc_id, chunks, vecs):
        """Оновлення документа = видалити старі чанки + додати нові."""
        self.delete_doc(doc_id)
        self.add(chunks, vecs)

    def compact(self):
        keep = [i for i, a in enumerate(self.alive) if a]
        chunks, vecs = [self.chunks[i] for i in keep], self.vecs[keep]
        self._reset()
        if chunks:
            self.add(chunks, vecs)
        self.compactions += 1

    def search(self, q, n):
        """Повертає до n внутрішніх id живих чанків, від найближчого."""
        total = self.index.ntotal
        if total == 0:
            return []
        kf = min(total, n + self.n_dead)
        self.index.hnsw.efSearch = max(self.ef_search, kf)
        _, ids = self.index.search(q[None, :], kf)
        return [int(i) for i in ids[0] if i >= 0 and self.alive[int(i)]][:n]


# ----------------------------------------------------------------------------
# Побудова й оцінювання
# ----------------------------------------------------------------------------
def load_chunks(root: Path, strategy):
    p = root / f"data/chunks/{strategy}.jsonl"
    if not p.exists():
        raise SystemExit(f"Немає {p}: спершу запустіть src/chunking.py")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_store(model, e5, chunks, args):
    t0 = time.perf_counter()
    vecs = embed(model, [c["text"] for c in chunks], "passage: " if e5 else "")
    t_emb = time.perf_counter() - t0
    store = HnswStore(vecs.shape[1], args.m, args.ef_construction, args.ef_search)
    t1 = time.perf_counter()
    store.add(chunks, vecs)
    t_idx = time.perf_counter() - t1
    return store, t_emb, t_idx


def evaluate(store, model, e5, queries, k):
    prefix = "query: " if e5 else ""
    store.search(embed(model, [queries[0]["q"]], prefix)[0], k)  # прогрів
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        qv = embed(model, [q["q"]], prefix)[0]
        t_emb = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        for _ in range(REPEAT):
            ids = store.search(qv, k * 6)  # запас: ієрархічні діти можуть мати одного батька
        t_srch = (time.perf_counter() - t1) * 1000 / REPEAT
        contexts, seen = [], set()
        for i in ids:
            c = store.chunks[i]
            key = c.get("parent_id") or c["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            contexts.append(c["context"])
            if len(contexts) == k:
                break
        rank = next((r for r, ctx in enumerate(contexts, 1)
                     if any(norm(e) in norm(ctx) for e in q["evidence"])), None)
        rows.append({"rank": rank, "emb_ms": t_emb, "srch_ms": t_srch})
    return rows


def summarize(rows):
    emb = np.array([r["emb_ms"] for r in rows])
    srch = np.array([r["srch_ms"] for r in rows])
    return {
        "recall": float(np.mean([r["rank"] is not None for r in rows])),
        "mrr": float(np.mean([1 / r["rank"] if r["rank"] else 0 for r in rows])),
        "emb_ms": float(emb.mean()), "emb_p95": float(np.percentile(emb, 95)),
        "srch_ms": float(srch.mean()), "srch_p95": float(np.percentile(srch, 95)),
    }


# ----------------------------------------------------------------------------
# Перевірка upsert / delete на n документах
# ----------------------------------------------------------------------------
def top_chunks(store, model, e5, text, n=5):
    qv = embed(model, [text], "query: " if e5 else "")[0]
    return [store.chunks[i] for i in store.search(qv, n)]


def crud_test(store, model, e5, doc_ids):
    """Для кожного документа: upsert зміненого тексту → delete → відновлення оригіналу."""
    passage = "passage: " if e5 else ""
    rows = []
    for d in doc_ids:
        old = store.chunks_of(d)
        old_ids = {c["chunk_id"] for c in old}
        probe = old[0]["text"][:300]
        pre_ok = any(c["doc_id"] == d for c in top_chunks(store, model, e5, probe))
        n0 = store.n_alive

        new = []
        for c in old:
            c2 = dict(c)
            c2["text"] = f"[ОНОВЛЕНО {d}] " + c["text"]
            c2["context"] = c2["text"]
            c2["chunk_id"] = c["chunk_id"] + "~upd"
            new.append(c2)
        new_vecs = embed(model, [c["text"] for c in new], passage)
        t0 = time.perf_counter()
        store.upsert_doc(d, new, new_vecs)
        t_up = (time.perf_counter() - t0) * 1000
        res = top_chunks(store, model, e5, new[0]["text"][:300])
        up_ok = (bool(res) and res[0]["chunk_id"] == new[0]["chunk_id"]
                 and not any(c["chunk_id"] in old_ids for c in res)
                 and store.n_alive == n0 - len(old) + len(new))

        t0 = time.perf_counter()
        store.delete_doc(d)
        t_del = (time.perf_counter() - t0) * 1000
        del_ok = (not any(c["doc_id"] == d for c in top_chunks(store, model, e5, probe))
                  and store.n_alive == n0 - len(old))

        old_vecs = embed(model, [c["text"] for c in old], passage)
        t0 = time.perf_counter()
        store.upsert_doc(d, old, old_vecs)
        t_rest = (time.perf_counter() - t0) * 1000
        rest_ok = (any(c["doc_id"] == d for c in top_chunks(store, model, e5, probe))
                   and store.n_alive == n0)
        rows.append({"doc": d, "n_old": len(old), "n_new": len(new), "pre": pre_ok,
                     "up": up_ok, "del": del_ok, "rest": rest_ok,
                     "t_up": t_up, "t_del": t_del, "t_rest": t_rest})
    return rows


# ----------------------------------------------------------------------------
# Звіт
# ----------------------------------------------------------------------------
def ok(b):
    return "✓" if b else "✗"


def write_report(root, args, n_chunks, n_queries, builds, evals, main_ef, per_query,
                 crud, compactions, bad, versions):
    L = [
        "# Порівняння моделей embeddings (FAISS HNSW)", "",
        f"Стратегія chunking: `{args.strategy}`; чанків: {n_chunks}; запитів: {n_queries}. "
        f"Метрика recall@{args.k} — частка запитів, у топ-{args.k} контекстів яких є еталонна фраза. "
        f"Версії: {versions}.", "",
        "Таблиця 1 — Параметри індексу (однакові для всіх моделей)", "",
        "| Параметр | Значення |", "|---|---|",
        f"| Тип індексу | IndexHNSWFlat, метрика — скалярний добуток нормованих векторів (косинус) |",
        f"| M | {args.m} |", f"| efConstruction | {args.ef_construction} |",
        f"| efSearch (основний) | {main_ef} |",
        f"| Значення efSearch у порівнянні | {', '.join(map(str, sorted({e for b in evals.values() for e in b})))} |",
        f"| Повторів пошуку для усереднення затримки | {REPEAT} |", "",
        "Таблиця 2 — Побудова", "",
        "| Модель | Вимір | Embedding чанків, с | Побудова HNSW, с | Вектори, МБ |",
        "|---|---|---|---|---|",
    ]
    for name, b in builds.items():
        L.append(f"| `{name}` | {b['dim']} | {b['t_emb']:.2f} | {b['t_idx']:.3f} | {b['mb']:.2f} |")
    L += ["", f"Таблиця 3 — Якість і затримка ({n_queries} запитів)", "",
          "| Модель | efSearch | recall@5 | MRR@5 | Embedding запиту, мс (сер. / p95) | "
          "Пошук, мс (сер. / p95) |", "|---|---|---|---|---|---|"]
    for name, by_ef in evals.items():
        for ef, s in sorted(by_ef.items()):
            mark = " **←**" if ef == main_ef else ""
            L.append(f"| `{name}` | {ef}{mark} | {s['recall']:.2f} | {s['mrr']:.2f} | "
                     f"{s['emb_ms']:.1f} / {s['emb_p95']:.1f} | {s['srch_ms']:.3f} / {s['srch_p95']:.3f} |")
    names = list(per_query)
    L += ["", f"Таблиця 4 — Позиція першого влучання в топ-{args.k} (efSearch = {main_ef}; «—» якщо немає)", "",
          "| № | Запит | " + " | ".join(f"`{n.split('/')[-1]}`" for n in names) + " |",
          "|---|---|" + "---|" * len(names)]
    for qi, q in enumerate(args.queries_loaded):
        cells = [str(per_query[n][qi]["rank"] or "—") for n in names]
        L.append(f"| {qi + 1} | {q['q']} | " + " | ".join(cells) + " |")
    if bad:
        L += ["", "Запити, виключені з оцінки (evidence не знайдено в корпусі):", ""]
        L += [f"- рядок {ln}: «{q}» — {m}" for ln, q, m in bad]

    if crud:
        L += ["", "## Оновлення індексу: upsert / delete", "",
              "Процедура:", "",
              "1. **delete(doc_id)** — усі чанки документа позначаються видаленими («надгробки»); "
              "HNSW у FAISS не вміє видаляти вузли, тому пошук бере з запасом (`n + кількість видалених`) "
              "і відфільтровує мертві.",
              "2. **upsert(doc_id, чанки)** — delete старих чанків + додавання нових векторів у HNSW.",
              f"3. **compact()** — коли видалених більше {int(DEAD_RATIO_LIMIT * 100)} %, індекс перебудовується "
              f"лише з живих векторів (під час перевірки спрацювало разів: {compactions}).",
              "4. Щоб не перераховувати зайве, перед upsert порівнюють хеш тексту документа з попереднім; "
              "ембедяться лише змінені документи.", "",
              "Таблиця 5 — Перевірка на 5 документах (перша модель). pre — чанк знаходить сам себе до змін; "
              "upsert — новий текст на 1-й позиції, старі чанки не повертаються, лічильник сходиться; "
              "delete — документ не з'являється у видачі; restore — оригінал повернуто.", "",
              "| Документ | Чанків (було → стало) | pre | upsert | delete | restore | "
              "t upsert, мс | t delete, мс | t restore, мс |", "|---|---|---|---|---|---|---|---|---|"]
        for r in crud:
            L.append(f"| {r['doc']} | {r['n_old']} → {r['n_new']} | {ok(r['pre'])} | {ok(r['up'])} | "
                     f"{ok(r['del'])} | {ok(r['rest'])} | {r['t_up']:.1f} | {r['t_del']:.2f} | {r['t_rest']:.1f} |")
    L += ["", "## Мова", "",
          "> TODO: корпус і запити українською. Поясніть, чим відрізняються моделі щодо мовної підтримки "
          "(multilingual-e5 навчена на багатомовних даних із префіксами `query:`/`passage:`; "
          "paraphrase-multilingual-MiniLM — на парафразах, без префіксів), і що це дає на ваших запитах.",
          "", "## Вибір і обґрунтування", "",
          "> TODO: власними словами оберіть модель, спираючись на таблиці 3–4 (recall, MRR, затримка, вимір, "
          "розмір), а також на вплив efSearch. Обмеження: 20 запитів, невеликий корпус.", ""]
    path = root / "docs/embeddings_cmp.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--strategy", default="structure", choices=["fixed", "structure", "hierarchical"],
                    help="стратегія chunking, обрана в Результаті 2")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="моделі sentence-transformers або 'hash'")
    ap.add_argument("--queries", default="eval/index_queries.jsonl")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--ef-sweep", default="16,64,128", help="значення efSearch для порівняння")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-docs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)
    random.seed(args.seed)
    np.random.seed(args.seed)

    chunks = load_chunks(root, args.strategy)
    docs = load_docs(root)
    corpus_norm = norm(" ".join(t for _, _, t in docs))
    qpath = root / args.queries
    if not qpath.exists():
        raise SystemExit(f"Немає {qpath}: потрібні 20 запитів у форматі {{\"q\":..., \"evidence\":[...]}}")
    queries, bad = load_queries(qpath, corpus_norm)
    for ln, q, m in bad:
        print(f"[warn] запит у рядку {ln} виключено: evidence не знайдено в корпусі: {m}")
    if not queries:
        raise SystemExit("Немає придатних запитів")
    if len(queries) < 20:
        print(f"[warn] придатних запитів {len(queries)}, потрібно 20")
    args.queries_loaded = queries
    ef_list = sorted({int(x) for x in args.ef_sweep.split(",") if x.strip()} | {args.ef_search})
    print(f"Чанків ({args.strategy}): {len(chunks)}; запитів: {len(queries)}")

    builds, evals, per_query = {}, {}, {}
    crud, compactions = None, 0
    idx_dir = root / "data/index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for mi, name in enumerate(args.models):
        e5 = "e5" in name.lower()
        model = load_model(name)
        store, t_emb, t_idx = build_store(model, e5, chunks, args)
        builds[name] = {"dim": store.dim, "t_emb": t_emb, "t_idx": t_idx,
                        "mb": store.vecs.nbytes / 1e6}
        slug = re.sub(r"\W+", "_", name).strip("_")
        faiss.write_index(store.index, str(idx_dir / f"{slug}_{args.strategy}_hnsw.faiss"))
        (idx_dir / f"{slug}_{args.strategy}_ids.json").write_text(
            json.dumps([c["chunk_id"] for c in store.chunks], ensure_ascii=False), encoding="utf-8")
        evals[name] = {}
        for ef in ef_list:
            store.ef_search = ef
            rows = evaluate(store, model, e5, queries, args.k)
            evals[name][ef] = summarize(rows)
            if ef == args.ef_search:
                per_query[name] = rows
            s = evals[name][ef]
            print(f"{name.split('/')[-1]:45s} ef={ef:4d} recall@{args.k}={s['recall']:.2f} "
                  f"MRR={s['mrr']:.2f} emb={s['emb_ms']:.1f}мс пошук={s['srch_ms']:.3f}мс")
        store.ef_search = args.ef_search
        if mi == 0:  # перевірка upsert/delete на першій моделі
            ids = sorted({c["doc_id"] for c in chunks})
            sample = random.sample(ids, min(args.test_docs, len(ids)))
            crud = crud_test(store, model, e5, sample)
            compactions = store.compactions
            for r in crud:
                print(f"[crud] {r['doc']}: upsert {ok(r['up'])} delete {ok(r['del'])} restore {ok(r['rest'])}")

    try:
        import sentence_transformers
        versions = (f"sentence-transformers {sentence_transformers.__version__}, "
                    f"faiss {faiss.__version__}, numpy {np.__version__}")
    except ImportError:
        versions = f"faiss {faiss.__version__}, numpy {np.__version__}"
    path = write_report(root, args, len(chunks), len(queries), builds, evals, args.ef_search,
                        per_query, crud, compactions, bad, versions)
    print(f"\nЗвіт: {path}")


if __name__ == "__main__":
    main()