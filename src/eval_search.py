#!/usr/bin/env python3
"""ЛР 10, Результат 5: золотий набір і оцінювання пошуку.

Три команди:
  run           (за замовчуванням) оцінює 4 конфігурації (bm25, dense, hybrid, hybrid_rerank)
                на eval/gold.jsonl: recall@5/10, MRR, nDCG@10, затримка, вартість;
                окремо — запити без відповіді; пише docs/search_eval.md.
  export-kappa  готує eval/kappa_pool_A.csv і eval/kappa_pool_B.csv для двох розмітників
                (≥ 10 запитів, пул кандидатів із усіх конфігурацій) та eval/kappa_key.csv (авторозмітка).
  kappa         читає заповнені файли обох розмітників і рахує κ Коена → eval/kappa_result.json
                (потім `run` додасть його до звіту).

Формат gold.jsonl (по одному JSON на рядок):
  {"id": "g01", "type": "keyword|paraphrase|multistep|unanswerable", "q": "...",
   "evidence": ["дослівна фраза з корпусу", ...]}      # для unanswerable — []
Релевантний фрагмент = той, чий контекст містить хоча б одну evidence-фразу. Тож розмітка не
залежить від chunk_id і однакова для всіх стратегій chunking.

Метрики (лише для запитів з відповіддю):
  recall@k — частка evidence-фраз, що трапилися в топ-k контекстів (для 1 фрази це «влучив/ні»);
  MRR      — 1 / позиція першого релевантного контексту (у топ-10);
  nDCG@10  — бінарна релевантність, ідеал = min(кількість релевантних у корпусі, 10).

Запуск:
  python src/eval_search.py --model hash --reranker overlap       (швидка перевірка)
  python src/eval_search.py export-kappa --n 12
  python src/eval_search.py kappa --a eval/kappa_pool_A.csv --b eval/kappa_pool_B.csv
  python src/eval_search.py run --strategy structure
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from chunking import embed, load_docs, load_model, norm
except ImportError:  # файл раніше називався chuncking.py
    from chuncking import embed, load_docs, load_model, norm
from hybrid import MODES, Retriever, load_reranker, tokenize
from index import load_chunks

TYPES = ["keyword", "paraphrase", "multistep", "unanswerable"]
YES = {"1", "так", "yes", "y", "+", "true", "т"}
NO = {"0", "ні", "no", "n", "-", "false", "н"}


# ----------------------------------------------------------------------------
# Дані
# ----------------------------------------------------------------------------
def load_gold(path: Path, corpus_norm: str):
    items, bad = [], []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        g = json.loads(line)
        ev = g.get("evidence", [])
        g["answerable"] = bool(ev) and g.get("type") != "unanswerable"
        miss = [e for e in ev if norm(e) not in corpus_norm]
        if miss:
            bad.append((ln, g["q"], miss))
        else:
            items.append(g)
    return items, bad


def build_retriever(args):
    chunks = load_chunks(Path(args.root), args.strategy)
    model = load_model(args.model)
    e5 = "e5" in args.model.lower()
    R = Retriever(chunks, model, e5, load_reranker(args.reranker), args)
    R.norm_ctx = [norm(c["context"]) for c in chunks]
    return R


def top_indices(R, ids, k):
    """Індекси чанків у видачі без дублів за батьком (для ієрархічної стратегії)."""
    out, seen = [], set()
    for i in ids:
        c = R.chunks[i]
        key = c.get("parent_id") or c["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
        if len(out) == k:
            break
    return out


def relevant_total(R, evidence):
    """Скільки різних (за батьком) контекстів у корпусі містять хоча б одну evidence-фразу."""
    keys = set()
    for i, c in enumerate(R.chunks):
        if any(norm(e) in R.norm_ctx[i] for e in evidence):
            keys.add(c.get("parent_id") or c["chunk_id"])
    return len(keys)


def score_query(R, idx, evidence, r_total):
    flags = [[norm(e) in R.norm_ctx[i] for e in evidence] for i in idx]
    rel = [any(f) for f in flags]

    def cov(k):
        return sum(any(f[j] for f in flags[:k]) for j in range(len(evidence))) / len(evidence)

    rank = next((n + 1 for n, r in enumerate(rel) if r), None)
    dcg = sum(1 / np.log2(n + 2) for n, r in enumerate(rel[:10]) if r)
    ideal = sum(1 / np.log2(n + 2) for n in range(min(r_total, 10)))
    return {"r5": cov(5), "r10": cov(10), "rr": 1 / rank if rank else 0.0,
            "ndcg": dcg / ideal if ideal else 0.0, "rank": rank}


# ----------------------------------------------------------------------------
# κ Коена
# ----------------------------------------------------------------------------
def parse_label(s):
    s = (s or "").strip().lower()
    return 1 if s in YES else 0 if s in NO else None


def cohen_kappa(a, b):
    n = len(a)
    n11 = sum(x == 1 and y == 1 for x, y in zip(a, b))
    n00 = sum(x == 0 and y == 0 for x, y in zip(a, b))
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n00 + n01) * (n00 + n10)) / n ** 2
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {"kappa": kappa, "po": po, "pe": pe, "matrix": [[n00, n01], [n10, n11]], "n": n}


def interpret(k):
    for lim, text in [(0.0, "гірше за випадковість"), (0.2, "слабка згода"), (0.4, "помірна"),
                      (0.6, "суттєва"), (0.8, "висока"), (1.01, "майже повна")]:
        if k < lim or lim == 1.01:
            return text
    return ""


def read_labels(path, col):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["qid"], r["chunk_key"])] = parse_label(r.get(col))
    return out


def cmd_kappa(args):
    A, B = read_labels(args.a, "label_A"), read_labels(args.b, "label_B")
    keys = [k for k in A if k in B and A[k] is not None and B[k] is not None]
    skipped = len(A) - len(keys)
    if len(keys) < 10:
        raise SystemExit(f"Замало розмічених пар ({len(keys)})")
    res = cohen_kappa([A[k] for k in keys], [B[k] for k in keys])
    res["n_queries"] = len({k[0] for k in keys})
    res["skipped_pairs"] = skipped
    res["interpretation"] = interpret(res["kappa"])
    keyf = Path(args.key)
    if keyf.exists():
        auto = {}
        with open(keyf, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                auto[(r["qid"], r["chunk_key"])] = int(r["auto"])
        for name, L in (("A", A), ("B", B)):
            ks = [k for k in keys if k in auto]
            res[f"kappa_{name}_vs_evidence"] = cohen_kappa([L[k] for k in ks], [auto[k] for k in ks])["kappa"]
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"κ = {res['kappa']:.3f} ({res['interpretation']}); згода {res['po']:.2%}; "
          f"пар {res['n']}, запитів {res['n_queries']}, пропущено {skipped} → {args.out}")


def cmd_export_kappa(args):
    root = Path(args.root)
    R = build_retriever(args)
    docs = load_docs(root)
    gold, _ = load_gold(root / args.gold, norm(" ".join(t for _, _, t in docs)))
    answerable = [g for g in gold if g["answerable"]]
    if len(answerable) < args.n:
        raise SystemExit(f"Запитів з відповіддю лише {len(answerable)}")
    sample = random.Random(args.seed).sample(answerable, args.n)
    modes = [m for m in MODES if m != "hybrid_rerank" or R.reranker is not None]
    header = ["qid", "q", "chunk_key", "doc_id", "section", "context", "label_A", "label_B"]
    rows, key_rows = [], []
    for g in sample:
        pool = {}
        for m in modes:
            for i in top_indices(R, R.search(g["q"], m, None, 30), 5):
                c = R.chunks[i]
                pool.setdefault(c.get("parent_id") or c["chunk_id"], i)
        keys = sorted(pool)
        random.Random(args.seed).shuffle(keys)  # порядок без підказок про метод
        for key in keys:
            c = R.chunks[pool[key]]
            rows.append([g["id"], g["q"], key, c["doc_id"], c["section"],
                         " ".join(c["context"].split())[:1800], "", ""])
            key_rows.append([g["id"], key, int(any(norm(e) in R.norm_ctx[pool[key]] for e in g["evidence"]))])
    for name, col in (("A", 6), ("B", 7)):
        p = root / f"eval/kappa_pool_{name}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
        print(f"→ {p}")
    with open(root / args.key, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "chunk_key", "auto"])
        w.writerows(key_rows)
    print(f"→ {root / args.key}  (авторозмітка за evidence; розмітникам НЕ показувати)")
    print(f"Запитів: {len(sample)}, пар «запит—фрагмент»: {len(rows)}.\n"
          "Кожен розмітник самостійно ставить у своєму файлі в колонці label_A / label_B: "
          "1 — фрагмент містить відповідь на запит, 0 — ні. Не радьтеся між собою.")


# ----------------------------------------------------------------------------
# Оцінювання
# ----------------------------------------------------------------------------
def best_threshold(pos, neg):
    best = (0.0, None)
    for t in sorted(set(pos + neg)):
        acc = (sum(p >= t for p in pos) + sum(n < t for n in neg)) / (len(pos) + len(neg))
        if acc > best[0]:
            best = (acc, t)
    return best


def cmd_run(args):
    root = Path(args.root)
    docs = load_docs(root)
    corpus_norm = norm(" ".join(t for _, _, t in docs))
    gold, bad = load_gold(root / args.gold, corpus_norm)
    for ln, q, m in bad:
        print(f"[warn] запит у рядку {ln} виключено: evidence не знайдено в корпусі: {m}")
    n_ans = sum(g["answerable"] for g in gold)
    n_un = len(gold) - n_ans
    print(f"Запитів: {len(gold)} (з відповіддю {n_ans}, без відповіді {n_un})")
    if len(gold) < 40 or n_un < 5:
        print("[warn] потрібно ≥ 40 запитів, з них ≥ 5 без відповіді")

    R = build_retriever(args)
    modes = [m for m in MODES if m != "hybrid_rerank" or R.reranker is not None]
    for m in modes:
        R.search(gold[0]["q"], m, None, 10)  # прогрів

    lost = [(g["id"], e) for g in gold for e in g["evidence"]
            if not any(norm(e) in nc for nc in R.norm_ctx)]
    for gid, e in lost:
        print(f"[warn] {gid}: evidence розрізано межею чанка або відсутня в чанках: {e!r}")
    r_totals = {g["id"]: relevant_total(R, g["evidence"]) for g in gold if g["answerable"]}

    res = {m: {} for m in modes}
    for m in modes:
        for g in gold:
            t0 = time.perf_counter()
            ids = R.search(g["q"], m, None, 60)
            ms = (time.perf_counter() - t0) * 1000
            idx = top_indices(R, ids, 10)
            r = {"ms": ms}
            if g["answerable"]:
                r.update(score_query(R, idx, g["evidence"], r_totals[g["id"]]))
            res[m][g["id"]] = r

    ans = [g for g in gold if g["answerable"]]

    def agg(mode, subset, key):
        return float(np.mean([res[mode][g["id"]][key] for g in subset]))

    # --- запити без відповіді: сигнал «впевненості» ---
    def top_scores(g):
        qv = embed(R.model, [g["q"]], "query: " if R.e5 else "")[0]
        return float((R.store.vecs @ qv).max()), float(R.bm25.get_scores(tokenize(g["q"], R.stem)).max())
    sc = {g["id"]: top_scores(g) for g in gold}
    un = [g for g in gold if not g["answerable"]]
    thr = {}
    for j, name in enumerate(("dense (макс. косинус)", "BM25 (макс. оцінка)")):
        pos, neg = [sc[g["id"]][j] for g in ans], [sc[g["id"]][j] for g in un]
        if pos and neg:
            acc, t = best_threshold(pos, neg)
            thr[name] = (np.mean(pos), np.mean(neg), t, acc)

    # --- звіт ---
    kappa = None
    kp = root / args.kappa_file
    if kp.exists():
        kappa = json.loads(kp.read_text(encoding="utf-8"))
    L = ["# Оцінювання пошуку", "",
         f"Золотий набір: `{args.gold}`; стратегія chunking: `{args.strategy}`; чанків: {len(R.chunks)}; "
         f"embeddings: `{args.model}`; reranker: `{args.reranker}`.", "",
         "## 1. Золотий набір", "",
         f"Запитів: **{len(gold)}**, з відповіддю: {n_ans}, без відповіді: **{n_un}**. "
         "Релевантний фрагмент — той, чий контекст містить еталонну фразу (evidence); для багатокрокових запитів "
         "recall@k рахується як частка evidence-фраз, знайдених у топ-k.", "",
         "| Тип | Запитів |", "|---|---|"]
    cnt = Counter(g["type"] for g in gold)
    L += [f"| {t} | {cnt.get(t, 0)} |" for t in TYPES]
    if bad:
        L += ["", "Виключено (evidence не знайдено в корпусі): " + "; ".join(f"рядок {ln}" for ln, _, _ in bad)]
    if lost:
        L += ["", "Evidence, яка не потрапила цілою в жоден чанк (занижує всі метрики): "
              + "; ".join(f"{gid}" for gid, _ in lost)]

    L += ["", f"## 2. Метрики (запити з відповіддю, n = {n_ans})", "",
          "| Конфігурація | recall@5 | recall@10 | MRR | nDCG@10 | Затримка, мс (сер. / p95) | "
          "Embedding-викликів / запит | Пар для reranker / запит |", "|---|---|---|---|---|---|---|---|"]
    for m in modes:
        ms = np.array([res[m][g["id"]]["ms"] for g in gold])
        L.append(f"| {m} | {agg(m, ans, 'r5'):.2f} | {agg(m, ans, 'r10'):.2f} | {agg(m, ans, 'rr'):.2f} | "
                 f"{agg(m, ans, 'ndcg'):.2f} | {ms.mean():.1f} / {np.percentile(ms, 95):.1f} | "
                 f"{0 if m == 'bm25' else 1} | {args.rerank_in if m == 'hybrid_rerank' else 0} |")
    L += ["", "Затримка — повний цикл пошуку на CPU (включно з embedding запиту й reranker). "
          "Вартість: усі моделі локальні, тож грошова вартість запиту = 0; витрати — це час CPU та пам'ять "
          "(reranker додає найбільше).", "",
          f"### recall@5 за типом запиту", "",
          "| Конфігурація | " + " | ".join(t for t in TYPES if t != "unanswerable") + " |",
          "|---|" + "---|" * 3]
    for m in modes:
        cells = []
        for t in TYPES[:3]:
            sub = [g for g in ans if g["type"] == t]
            cells.append(f"{agg(m, sub, 'r5'):.2f} (n={len(sub)})" if sub else "—")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L += ["", f"## 3. Запити без відповіді (n = {n_un})", "",
          "Метрики recall/MRR/nDCG для них не визначені. Правильна поведінка системи — не вигадувати відповідь, "
          "тому перевіряємо, чи можна відрізнити такі запити за найкращою оцінкою відповідності.", "",
          "| Сигнал | Сер. для запитів із відповіддю | Сер. для запитів без відповіді | Найкращий поріг | "
          "Точність розрізнення |", "|---|---|---|---|---|"]
    for name, (p, n, t, acc) in thr.items():
        L.append(f"| {name} | {p:.3f} | {n:.3f} | {t:.3f} | {acc:.2f} |")
    L += ["", "> Висновок з 7 запитів — орієнтир, а не доказ: поріг підбирався на тих самих даних."]

    L += ["", "## 4. Узгодженість розмітки (κ Коена)", ""]
    if kappa:
        L += [f"Двоє розмітників оцінили {kappa['n']} пар «запит—фрагмент» для {kappa['n_queries']} запитів "
              "(незалежно, мітка «1» — фрагмент містить відповідь).", "",
              "| Показник | Значення |", "|---|---|",
              f"| κ Коена | **{kappa['kappa']:.3f}** ({kappa['interpretation']}) |",
              f"| Спостережувана згода | {kappa['po']:.2%} |", f"| Очікувана випадкова згода | {kappa['pe']:.2%} |"]
        for nm in ("A", "B"):
            if f"kappa_{nm}_vs_evidence" in kappa:
                L.append(f"| κ розмітника {nm} з авторозміткою за evidence | {kappa[f'kappa_{nm}_vs_evidence']:.3f} |")
        mtx = kappa["matrix"]
        L += ["", "Матриця (рядки — розмітник A, стовпці — B):", "",
              "| | B=0 | B=1 |", "|---|---|---|", f"| A=0 | {mtx[0][0]} | {mtx[0][1]} |",
              f"| A=1 | {mtx[1][0]} | {mtx[1][1]} |", "",
              "> TODO: опишіть, у яких випадках розмітники розійшлися і як це вплинуло на золотий набір."]
    else:
        L += ["> TODO: κ ще не обчислено. `python src/eval_search.py export-kappa`, двоє розмітників "
              "заповнюють свої CSV самостійно, потім `python src/eval_search.py kappa --a ... --b ...` "
              "і повторний запуск `run`."]

    best = max(modes, key=lambda m: (agg(m, ans, "ndcg"), agg(m, ans, "r5")))
    fast = min(modes, key=lambda m: np.mean([res[m][g["id"]]["ms"] for g in gold]))
    L += ["", "## 5. Висновок для ЛР 11", "",
          f"> Підказка за даними: найвищий nDCG@10 — **{best}**, найшвидша — **{fast}**.",
          "> TODO: власними словами оберіть конфігурацію для ЛР 11 і обґрунтуйте: різниця в якості між "
          "конфігураціями проти різниці в затримці та складності (reranker), поведінка на типах запитів "
          "і на запитах без відповіді, обмеження (малий корпус, один тип еталону, evidence-розмітка).", ""]
    path = root / "docs/search_eval.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    for m in modes:
        print(f"{m:14s} recall@5={agg(m, ans, 'r5'):.2f} recall@10={agg(m, ans, 'r10'):.2f} "
              f"MRR={agg(m, ans, 'rr'):.2f} nDCG@10={agg(m, ans, 'ndcg'):.2f}")
    print(f"\nЗвіт: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "export-kappa", "kappa"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--gold", default="eval/gold.jsonl")
    ap.add_argument("--strategy", default="structure", choices=["fixed", "structure", "hierarchical"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--reranker", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rerank-in", type=int, default=30)
    ap.add_argument("--stem", type=int, default=6)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--n", type=int, default=12, help="export-kappa: скільки запитів розмічати (≥ 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--key", default="eval/kappa_key.csv")
    ap.add_argument("--a", default="eval/kappa_pool_A.csv")
    ap.add_argument("--b", default="eval/kappa_pool_B.csv")
    ap.add_argument("--out", default="eval/kappa_result.json")
    ap.add_argument("--kappa-file", default="eval/kappa_result.json")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    {"run": cmd_run, "export-kappa": cmd_export_kappa, "kappa": cmd_kappa}[args.cmd](args)


if __name__ == "__main__":
    main()