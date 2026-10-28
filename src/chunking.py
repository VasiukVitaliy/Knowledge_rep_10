import argparse
import hashlib
import json
import re
import time
from datac import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ITEM_RE = re.compile(r"^(\d{1,2}(?:\.\d+)+)\.?\s") 
HEAD_RE = re.compile(r"^(?:#{1,6}\s+.+|\d{1,2}\.\s+[А-ЯІЇЄҐA-Z].{2,100})$") 
SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    section: str
    text: str              
    context: str                 
    parent_id: Optional[str]
    title: str
    type: str
    date: str
    sensitivity: str
    source_url: str


def split_long(line, limit):
    if len(line) <= limit:
        return [line]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;:!?])\s+", line):
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    out = []
    for p in parts:  
        while len(p) > limit:
            out.append(p[:limit])
            p = p[limit:]
        out.append(p)
    return out


def chunk_fixed(text, size=800, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start + int(size * 0.6):end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind("? "), window.rfind("! "))
            if cut >= 0:
                end = start + int(size * 0.6) + cut + 1
            else:
                sp = text.rfind(" ", start + 1, end)
                if sp > start:
                    end = sp
        piece = text[start:end].strip()
        if piece:
            out.append((f"символи {start}–{end}", piece))
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        while nxt < n and nxt > 0 and text[nxt - 1] != " ":  
            nxt += 1
        start = nxt
    return out


def chunk_structure(text, min_chars=300, max_chars=1200):
    units = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = ITEM_RE.match(line)
        head = (not item) and bool(HEAD_RE.match(line))
        for piece in split_long(line, max_chars):
            units.append((item.group(1) if item else None, head, bool(item), piece))

    chunks, buf = [], []
    state = {"size": 0, "first": None, "last": None, "head": None}
    cur_head = None

    def flush():
        if not buf:
            return
        base = state["head"] or ""
        if state["first"] and state["last"] and state["first"] != state["last"]:
            sec = f"{base}, п. {state['first']}–{state['last']}"
        elif state["first"]:
            sec = f"{base}, п. {state['first']}"
        else:
            sec = base
        chunks.append((sec.strip(", "), "\n".join(buf)))
        buf.clear()
        state.update(size=0, first=None, last=None)

    for label, is_head, is_item, piece in units:
        if buf and (is_head or state["size"] + len(piece) > max_chars
                    or (is_item and state["size"] >= min_chars)):
            flush()
        if is_head:
            cur_head = piece[:100]
        if not buf:
            state["head"] = cur_head
            state["first"] = label
        state["last"] = label or state["last"]
        buf.append(piece)
        state["size"] += len(piece) + 1
    flush()

    if len(chunks) > 1 and len(chunks[-1][1]) < min_chars:  
        sec, txt = chunks.pop()
        psec, ptxt = chunks.pop()
        chunks.append((psec, ptxt + "\n" + txt))
    return chunks


def chunk_hierarchical(text, parent_max=1800, child_size=350):
    out = []
    for pi, (section, ptext) in enumerate(chunk_structure(text, min_chars=500, max_chars=parent_max)):
        sents = [s.strip() for s in SENT_SPLIT.split(ptext) if s.strip()]
        cur = ""
        for s in sents:
            for piece in split_long(s, child_size):
                if cur and len(cur) + len(piece) + 1 > child_size:
                    out.append((section, cur, pi, ptext))
                    cur = piece
                else:
                    cur = f"{cur} {piece}".strip()
        if cur:
            out.append((section, cur, pi, ptext))
    return out


def load_docs(root: Path):
    meta = {}
    with (root / "data/corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta.setdefault(r["doc_id"], {k: r[k] for k in
                                        ("title", "type", "date", "sensitivity", "source_url")})
    docs = []
    for doc_id, m in meta.items():
        p = root / "data/clean" / f"{doc_id}.txt"
        if p.exists():
            docs.append((doc_id, m, p.read_text(encoding="utf-8")))
        else:
            print(f"[warn] немає {p} — документ пропущено")
    return docs


def build_chunks(strategy, docs, params):
    chunks = []
    for doc_id, m, text in docs:
        if strategy == "fixed":
            rows = [(s, t, t, None) for s, t in chunk_fixed(text, params["fixed_size"], params["fixed_overlap"])]
        elif strategy == "structure":
            rows = [(s, t, t, None) for s, t in chunk_structure(text, max_chars=params["struct_max"])]
        elif strategy == "hierarchical":
            rows = [(s, c, p, f"{doc_id}:p{pi}") for s, c, pi, p in
                    chunk_hierarchical(text, params["parent_max"], params["child_size"])]
        else:
            raise ValueError(strategy)
        for i, (section, ctext, context, parent) in enumerate(rows, 1):
            chunks.append(Chunk(f"{doc_id}#{strategy[:3]}{i:03d}", doc_id, strategy, section, ctext,
                                context, parent, m["title"], m["type"], m["date"],
                                m["sensitivity"], m["source_url"]))
    return chunks


class HashEmbedder:
    dim = 512

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.casefold()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


def load_model(name):
    if name == "hash":
        return HashEmbedder()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed(model, texts, prefix=""):
    v = model.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(v, dtype="float32")


class FlatIndex:

    def __init__(self, vecs):
        self.vecs = vecs
        try:
            import faiss
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
        except ImportError:
            self.index = None

    def search(self, q, k):
        k = min(k, len(self.vecs))
        if self.index is not None:
            _, ids = self.index.search(q[None, :], k)
            return [int(i) for i in ids[0] if i >= 0]
        return [int(i) for i in np.argsort(-(self.vecs @ q))[:k]]


def norm(s):
    return re.sub(r"\s+", " ", s).casefold()


def load_queries(path: Path, corpus_norm: str):
    qs, bad = [], []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        ev = [e for e in q["evidence"]]
        missing = [e for e in ev if norm(e) not in corpus_norm]
        if missing:
            bad.append((ln, q["q"], missing))
        else:
            qs.append(q)
    return qs, bad


def evaluate(chunks, index, q_vecs, queries, k=5):
    rows = []
    for qi, q in enumerate(queries):
        ids = index.search(q_vecs[qi], k * 6 if chunks[0].parent_id else k)
        contexts, seen = [], set()
        for i in ids:
            c = chunks[i]
            key = c.parent_id or c.chunk_id
            if key in seen:
                continue
            seen.add(key)
            contexts.append(c.context)
            if len(contexts) == k:
                break
        rank = next((r for r, ctx in enumerate(contexts, 1)
                     if any(norm(e) in norm(ctx) for e in q["evidence"])), None)
        rows.append({"hit": rank is not None, "rank": rank,
                     "context_chars": sum(len(c) for c in contexts)})
    return rows


def write_report(root, model_name, params, stats, per_query, queries, bad, versions):
    L = [
        "# Порівняння стратегій chunking", "",
        f"Модель embeddings (однакова для всіх стратегій): `{model_name}`. "
        f"Версії: {versions}. Запитів: {len(queries)}. Метрика recall@5 — частка запитів, "
        "у топ-5 контекстів яких є еталонна фраза (evidence).", "",
        "Таблиця 1 — Параметри стратегій", "",
        "| Стратегія | Параметри |", "|---|---|",
        f"| fixed | розмір {params['fixed_size']} символів, overlap {params['fixed_overlap']} |",
        f"| structure | за заголовками й пунктами, макс. {params['struct_max']} символів |",
        f"| hierarchical | діти ≈ {params['child_size']} символів для пошуку, "
        f"батьки до {params['parent_max']} символів як контекст |", "",
        "Таблиця 2 — Порівняння", "",
        "| Стратегія | Чанків | Сер. розмір чанка, симв. | recall@5 | MRR@5 | "
        "Сер. розмір контексту (топ-5), симв. | Час embedding, с |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        L.append(f"| {name} | {s['n']} | {s['avg_chunk']:.0f} | {s['recall']:.2f} | {s['mrr']:.2f} | "
                 f"{s['avg_ctx']:.0f} | {s['t_embed']:.1f} |")
    L += ["", "Таблиця 3 — Результат по запитах (позиція першого влучання в топ-5, «—» якщо немає)", "",
          "| № | Запит | " + " | ".join(stats) + " |", "|---|---|" + "---|" * len(stats)]
    for qi, q in enumerate(queries):
        cells = [str(per_query[n][qi]["rank"] or "—") for n in stats]
        L.append(f"| {qi + 1} | {q['q']} | " + " | ".join(cells) + " |")
    if bad:
        L += ["", "Запити, виключені з оцінки (evidence не знайдено в корпусі):", ""]
        L += [f"- рядок {ln}: «{q}» — {m}" for ln, q, m in bad]
    best = max(stats, key=lambda n: (stats[n]["recall"], stats[n]["mrr"], -stats[n]["avg_ctx"]))
    L += ["", "## Вибір і обґрунтування", "",
          f"> Підказка за даними: найвищий recall@5 (далі MRR, потім менший контекст) у стратегії **{best}**.",
          "> TODO: власними словами поясніть вибір, спираючись на таблиці 2–3: баланс recall і розміру "
          "контексту, поведінка на запитах, де стратегії розійшлися, обмеження (10 запитів).", ""]
    path = root / "docs/chunking_cmp.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small",
                    help="модель sentence-transformers або 'hash' для швидкої перевірки")
    ap.add_argument("--queries", default="eval/chunk_queries.jsonl")
    ap.add_argument("--fixed-size", type=int, default=800)
    ap.add_argument("--fixed-overlap", type=int, default=150)
    ap.add_argument("--struct-max", type=int, default=1200)
    ap.add_argument("--child-size", type=int, default=350)
    ap.add_argument("--parent-max", type=int, default=1800)
    args = ap.parse_args()
    root = Path(args.root)
    params = {"fixed_size": args.fixed_size, "fixed_overlap": args.fixed_overlap,
              "struct_max": args.struct_max, "child_size": args.child_size,
              "parent_max": args.parent_max}

    docs = load_docs(root)
    if not docs:
        raise SystemExit("Немає документів: спершу запустіть build_corpus.py")
    print(f"Документів: {len(docs)}")

    strategies = ["fixed", "structure", "hierarchical"]
    all_chunks = {}
    for s in strategies:
        ch = build_chunks(s, docs, params)
        all_chunks[s] = ch
        out = root / f"data/chunks/{s}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in ch:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        avg = np.mean([len(c.text) for c in ch])
        print(f"{s:13s} чанків: {len(ch):4d}  сер. розмір: {avg:6.0f}")

    qpath = root / args.queries
    if not qpath.exists():
        print(f"\nНемає {qpath}. Створіть 10 запитів (формат — на початку файлу), "
              "потім запустіть ще раз для recall@5.")
        return
    corpus_norm = norm(" ".join(t for _, _, t in docs))
    queries, bad = load_queries(qpath, corpus_norm)
    for ln, q, m in bad:
        print(f"[warn] запит у рядку {ln} виключено: evidence не знайдено в корпусі: {m}")
    if not queries:
        raise SystemExit("Немає придатних запитів")
    if len(queries) < 10:
        print(f"[warn] придатних запитів {len(queries)}, потрібно 10")

    e5 = "e5" in args.model.lower()
    model = load_model(args.model)
    q_vecs = embed(model, [q["q"] for q in queries], "query: " if e5 else "")

    stats, per_query = {}, {}
    for s in strategies:
        ch = all_chunks[s]
        t0 = time.time()
        vecs = embed(model, [c.text for c in ch], "passage: " if e5 else "")
        t_embed = time.time() - t0
        index = FlatIndex(vecs)
        idx_dir = root / "data/index"
        idx_dir.mkdir(parents=True, exist_ok=True)
        np.save(idx_dir / f"{s}.npy", vecs)

        rows = evaluate(ch, index, q_vecs, queries)
        per_query[s] = rows
        stats[s] = {
            "n": len(ch),
            "avg_chunk": float(np.mean([len(c.text) for c in ch])),
            "recall": float(np.mean([r["hit"] for r in rows])),
            "mrr": float(np.mean([1 / r["rank"] if r["rank"] else 0 for r in rows])),
            "avg_ctx": float(np.mean([r["context_chars"] for r in rows])),
            "t_embed": t_embed,
        }
        print(f"{s:13s} recall@5={stats[s]['recall']:.2f}  MRR={stats[s]['mrr']:.2f}  "
              f"контекст={stats[s]['avg_ctx']:.0f}")

    try:
        import sentence_transformers
        versions = f"sentence-transformers {sentence_transformers.__version__}, numpy {np.__version__}"
    except ImportError:
        versions = f"numpy {np.__version__}"
    path = write_report(root, args.model, params, stats, per_query, queries, bad, versions)
    print(f"\nЗвіт: {path}")


if __name__ == "__main__":
    main()