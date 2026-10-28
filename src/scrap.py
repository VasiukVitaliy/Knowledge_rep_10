import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

PDR_URL = "https://zakon.rada.gov.ua/laws/show/1306-2001-%D0%BF/print"

PDR_CHAPTERS = {
    "pdr_01_zagalni_polozhennya": (1, "Загальні положення"),
    "pdr_02_obovyazky_vodiyiv": (2, "Обов’язки і права водіїв механічних транспортних засобів"),
    "pdr_08_regulyuvannya": (8, "Регулювання дорожнього руху"),
    "pdr_10_pochatok_rukhu": (10, "Початок руху та зміна його напрямку"),
    "pdr_11_roztashuvannya": (11, "Розташування транспортних засобів на дорозі"),
    "pdr_12_shvydkist": (12, "Швидкість руху"),
    "pdr_13_dystantsiya": (13, "Дистанція, інтервал, зустрічний роз’їзд"),
    "pdr_14_obgin": (14, "Обгін"),
    "pdr_15_zupynka": (15, "Зупинка і стоянка"),
    "pdr_16_perekhrestya": (16, "Проїзд перехресть"),
    "pdr_17_perevahy_marshrutnykh": (17, "Переваги маршрутних транспортних засобів"),
    "pdr_18_pishokhidni_perekhody": (18, "Проїзд пішохідних переходів і зупинок транспортних засобів"),
}

SECTION_RULES = {doc_id: n for doc_id, (n, _) in PDR_CHAPTERS.items()}

SOURCES = {
    doc_id: {
        "title": f"ПДР України — Розділ {n}. {hint}",
        "url": PDR_URL,
        "type": "нормативний",
        "date": "2026-10-14",
        "sensitivity": "public",
    }
    for doc_id, (n, hint) in PDR_CHAPTERS.items()
}

SOURCES.update({
    "dtp_stat_2024_mvs": {
        "title": "Статистика ДТП за 2024 рік — МВС України",
        "url": "https://mvs.gov.ua/news/u-mvs-ozvucili-statistiku-dtp-za-2024-rik",
        "type": "статистика",
        "date": "2025-01-30",
        "sensitivity": "public"
    },
    "dtp_stat_interactive": {
        "title": "Інтерактивна статистика ДТП",
        "url": "https://testpdr.online/statystyka-dtp/",
        "type": "статистика",
        "date": "2026-01-01",
        "sensitivity": "restricted"
    },
    "lviv_smart_prio": {
        "title": "Як працює Smart Prio на світлофорах у Львові",
        "url": "https://dyvys.info/2026/09/07/smart-prio-na-svitloforah-u-lvovi/",
        "type": "новина",
        "date": "2026-09-07",
        "sensitivity": "restricted"
    },
    "lviv_svitloforni_rezhymy": {
        "title": "Світлофорні режими у Львові: смарт, зелена хвиля, адаптив",
        "url": "https://dyvys.info/2026/03/11/svitloforni-rezhymy-u-lvovi-de-pratsyuyut/",
        "type": "новина",
        "date": "2026-03-11",
        "sensitivity": "restricted"
    },
    "lviv_sykhiv_adaptive": {
        "title": "Адаптивне регулювання світлофорів на Сихові",
        "url": "https://tvoemisto.tv/uk/news/628-u-lvovi-na-sykhovi-zaprovadyly-adaptyvne-rehulyuvannya-svitloforiv-shcho-tse-oznachaye",
        "type": "новина",
        "date": "2026-03-30",
        "sensitivity": "restricted"
    },
    "lviv_lychakivska": {
        "title": "Модернізація світлофорів на вул. Личаківській",
        "url": "https://zahid.espreso.tv/novyny-lviv-na-dilyantsi-na-lichakivskiy-u-lvovi-modernizuvali-svitlofori",
        "type": "новина",
        "date": "2025-01-17",
        "sensitivity": "restricted"
    },
    "lviv_shevchenka_bryukhovychi": {
        "title": "Нові світлофори на перехресті Шевченка — Брюховицької",
        "url": "https://suspilne.media/lviv/1400046-na-perehresti-vulic-sevcenka-ta-bruhovickoi-u-lvovi-zapracuut-svitlofori/",
        "type": "новина",
        "date": "2026-09-10",
        "sensitivity": "restricted"
    },
    "lviv_olhy_boychuka": {
        "title": "Нові світлофори на перехресті Княгині Ольги — Бойчука",
        "url": "https://suspilne.media/lviv/1034507-vid-5-cervna-na-perehresti-vulic-knagini-olgi-ta-bojcuka-zapracuut-novi-svitlofori-so-vidomo/",
        "type": "новина",
        "date": "2025-06-04",
        "sensitivity": "restricted"
    },
    "lvivavtodor_adaptive_concept": {
        "title": "Концепція розвитку адаптивного регулювання світлофорів у Львові",
        "url": "https://lvivavtodor.com.ua/%D0%BD%D0%B0%D0%B9%D1%86%D1%96%D0%BA%D0%B0%D0%B2%D1%96%D1%88%D0%B5-%D0%BF%D1%80%D0%BE-%D0%B0%D0%B4%D0%B0%D0%BF%D1%82%D0%B8%D0%B2%D0%BD%D0%B5-%D1%81%D0%B2%D1%96%D1%82%D0%BB%D0%BE%D1%84%D0%BE%D1%80/",
        "type": "офіційний",
        "date": "2023-05-19",
        "sensitivity": "public"
    },
    "zaxid_dai_cameras_2011": {
        "title": "У Львові придумали, як позбутися корків (відеокамери й центр управління рухом)",
        "url": "https://zaxid.net/u_lvovi_pridumali_yak_pozbutisya_korkiv_n1128157",
        "type": "новина", "date": "2011-04-29", "sensitivity": "restricted", "optional": True
    },
    "zaxid_new_lights_centre_2012": {
        "title": "У центрі Львова завтра включать нові світлофори",
        "url": "https://zaxid.net/u_tsentri_lvova_zavtra_vklyuchat_novi_svitlofori_n1264278",
        "type": "новина", "date": "2012-09-05", "sensitivity": "restricted", "optional": True
    },
    "zaxid_transport_problems_2016": {
        "title": "Як вирішити транспортні проблеми Львова. По пунктах",
        "url": "https://zaxid.net/yak_virishiti_transportni_problemi_lvova_po_punktah_n1406419",
        "type": "блог", "date": "2016-10-12", "sensitivity": "restricted", "optional": True
    },
    "zaxid_transport_plan_2020": {
        "title": "У Львові затвердили план розвитку транспорту до 2020 року",
        "url": "https://zaxid.net/u_lvovi_zatverdili_plan_rozvitku_transportu_do_2020_roku_n1459220",
        "type": "новина", "date": "2018-07-06", "sensitivity": "restricted", "optional": True
    },
})

LICENSE_BY_DOMAIN = {
    "zakon.rada.gov.ua": "публічне надбання (офіційний акт, ст. 10 ЗУ «Про авторське право і суміжні права»)",
    "dyvys.info": "усі права захищено; використано в навчальних цілях",
    "tvoemisto.tv": "усі права захищено; використано в навчальних цілях",
    "zahid.espreso.tv": "усі права захищено; використано в навчальних цілях",
    "suspilne.media": "усі права захищено; використано в навчальних цілях",
    "zaxid.net": "усі права захищено; використано в навчальних цілях",
}
DEFAULT_LICENSE = "ліцензію не вказано; використано в навчальних цілях"
CC_RE = re.compile(r"creativecommons\.org/(?:licenses|publicdomain)/([a-z\-]+)/?([\d.]*)", re.I)


def guess_license(url, html_bytes):
    if html_bytes:
        m = CC_RE.search(html_bytes.decode("utf-8", errors="ignore"))
        if m:
            return f"CC {m.group(1).upper()} {m.group(2)}".strip(), "автовизначено на сторінці, перевірте"
    domain = urlparse(url).netloc
    if domain in LICENSE_BY_DOMAIN:
        return LICENSE_BY_DOMAIN[domain], "за доменом, перевірте"
    return DEFAULT_LICENSE, "за замовчуванням, перевірте"


NAV_LINE = re.compile(
    r"^(Попередній|Наступний|Версія для друку|Друкувати|Додати до закладок|Закладки|"
    r"Поділитися|Редакції|Зміст|Нагору|Вгору|Головна|Перейти до .*|Підписатися.*)$",
    re.I,
)
TEXT_RULES = [
    ("nbsp", re.compile(r"[\u00a0\u2009\u200b\ufeff]"), " ",
     "нерозривні та нульової ширини пробіли → звичайний пробіл"),
    ("amend_notes", re.compile(r"\{[^{}]*\}", re.S), "",
     "примітки про зміни {Пункт ... із змінами, внесеними згідно з ...} видалено"),
    ("multi_space", re.compile(r"[ \t]{2,}"), " ", "кілька пробілів → один"),
]

SHELL_MARKERS = ("Ваш броузер застарів", "Потрібно включити javascript")

MIN_CHARS, TARGET_CHARS, MAX_CHARS = 130, 380, 700
MIN_CHAPTER_CHARS = 800
ITEM_RE = re.compile(r"^(\d{1,2}(?:\.\d+)+)\.?\s")
HEADING_RE = re.compile(r"^(\d{1,2})\.\s+([А-ЯІЇЄҐA-Z][^\n]{2,100})$", re.M)


def fetch(url, cache_dir: Path, refresh=False, delay=1.0):
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    if not refresh:
        for ext in ("html", "pdf"):
            p = cache_dir / f"{key}.{ext}"
            if p.exists():
                return p.read_bytes(), ext
    import requests
    r = requests.get(url, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0 (lab10-student; educational use)"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "").lower()
    kind = "pdf" if ("pdf" in ctype or url.lower().endswith(".pdf")) else "html"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = r.content
    (cache_dir / f"{key}.{kind}").write_bytes(data)
    time.sleep(delay)
    return data, kind


def rada_to_text(html: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        t.decompose()
    node = soup.find(id="article") or soup.body or soup
    paras = []
    for p in node.find_all("p"):
        s = re.sub(r"\s+", " ", p.get_text(" ")).strip()
        s = re.sub(r"\s+([,.;:!?)»])", r"\1", s)
        s = re.sub(r"([(«])\s+", r"\1", s)
        if s:
            paras.append(s)
    if len(paras) > 100:
        return "\n".join(paras)
    return node.get_text("\n")


def html_to_text(data: bytes, url: str = ""):
    html = data.decode("utf-8", errors="replace")
    if urlparse(url).netloc == "zakon.rada.gov.ua":
        return rada_to_text(html)
    import trafilatura
    text = trafilatura.extract(html, include_tables=True, include_comments=False,
                             favor_recall=True)
    if text and len(text) > 500:
        return text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "form"]):
        t.decompose()
    node = soup.find(id="article") or soup.body or soup
    return node.get_text("\n")


def pdf_to_text(data: bytes):
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [p.get_text("text") for p in doc]
    freq = Counter()
    for pg in pages:
        for ln in {l.strip() for l in pg.splitlines() if l.strip()}:
            freq[ln] += 1
    limit = max(3, int(0.3 * len(pages)))
    repeated = {ln for ln, c in freq.items() if c > limit and len(ln) < 120}
    out = []
    for pg in pages:
        out.append("\n".join(l for l in pg.splitlines() if l.strip() not in repeated))
    return "\n".join(out), len(repeated)


def clean(text: str, stats: Counter):
    for name, rx, repl, _ in TEXT_RULES:
        found = len(rx.findall(text))
        if found:
            stats[name] += found
            text = rx.sub(repl, text)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and NAV_LINE.match(ln):
            stats["nav_lines"] += 1
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def cut_chapter(text: str, n: int):
    def is_caps(t):
        letters = [c for c in t if c.isalpha()]
        return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8
    heads = [(int(m.group(1)), m.start(), m.group(2).strip()) for m in HEADING_RE.finditer(text)]
    starts = [(pos, title) for num, pos, title in heads if num == n and is_caps(title)]
    ends = [pos for num, pos, title in heads if num == n + 1 and is_caps(title)]
    best = None
    for s, title in starts:
        e = next((x for x in ends if x > s), None)
        if e is None:
            continue
        if best is None or (e - s) > (best[1] - best[0]):
            best = (s, e, title)
    if best is None:
        return None, None
    return text[best[0]:best[1]].strip(), f"{n}. {best[2]}"


def split_long(line: str, limit: int):
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
    return parts


def make_chunks(text: str, base_section: str):
    units, cur_item = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = ITEM_RE.match(line)
        if m:
            cur_item = m.group(1)
        for piece in split_long(line, MAX_CHARS):
            units.append((cur_item, piece, bool(m)))

    chunks, buf, size, first, last = [], [], 0, None, None

    def flush():
        nonlocal buf, size, first, last
        if not buf:
            return
        if first and last and first != last:
            sec = f"{base_section}, п. {first}–{last}"
        elif first:
            sec = f"{base_section}, п. {first}"
        else:
            sec = base_section
        chunks.append((sec, "\n".join(buf)))
        buf, size, first, last = [], 0, None, None

    for item, piece, is_item_start in units:
        if buf and (size + len(piece) > MAX_CHARS or (size >= TARGET_CHARS and is_item_start)):
            flush()
        if not buf:
            first = item
        last = item or last
        buf.append(piece)
        size += len(piece) + 1
    flush()

    if len(chunks) > 1 and len(chunks[-1][1]) < MIN_CHARS:
        sec, txt = chunks.pop()
        psec, ptxt = chunks.pop()
        chunks.append((psec, ptxt + "\n" + txt))
    return chunks


def process_source(doc_id, meta, root: Path, refresh, stats: Counter):
    data, kind = fetch(meta["url"], root / "data/raw", refresh)

    if not meta.get("license"):
        lic, how = guess_license(meta["url"], data if kind == "html" else None)
        meta["license"], meta["license_how"] = lic, how

    if kind == "pdf":
        text, n_rep = pdf_to_text(data)
        stats["pdf_header_footer_lines"] += n_rep
    else:
        text = html_to_text(data, meta["url"])
    text = clean(text, stats)

    if len(text) < 1000 or (any(m in text for m in SHELL_MARKERS) and len(text) < 20000):
        raise RuntimeError("замість тексту документа отримано оболонку сторінки")

    base_section = meta["title"]
    if doc_id in SECTION_RULES:
        part, heading = cut_chapter(text, SECTION_RULES[doc_id])
        if part is None:
            raise RuntimeError(f"не знайдено розділ {SECTION_RULES[doc_id]}")
        if len(part) < MIN_CHAPTER_CHARS:
            raise RuntimeError(f"розділ {SECTION_RULES[doc_id]} надто короткий")
        text, base_section = part, heading
        meta["title"] = f"ПДР України — Розділ {heading}"

    clean_dir = root / "data/clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    records, seen = [], set()
    for i, (section, chunk_text) in enumerate(make_chunks(text, base_section), 1):
        h = hashlib.md5(chunk_text.encode()).hexdigest()
        if h in seen:
            stats["duplicate_chunks"] += 1
            continue
        seen.add(h)
        records.append({
            "chunk_id": f"{doc_id}#{i:03d}",
            "doc_id": doc_id,
            "title": meta["title"],
            "section": section,
            "type": meta["type"],
            "date": meta["date"],
            "accessed": date.today().isoformat(),
            "sensitivity": meta["sensitivity"],
            "restricted": meta["sensitivity"] != "public",
            "source_url": meta["url"],
            "lang": meta.get("lang", "uk"),
            "license": meta["license"],
            "n_chars": len(chunk_text),
            "text": chunk_text,
        })
    return records


def write_card(root: Path, all_records, stats: Counter, errors, force):
    by_doc = Counter(r["doc_id"] for r in all_records)
    sens = Counter(r["sensitivity"] for r in all_records)
    langs = Counter(r["lang"] for r in all_records)
    lines = [
        "# Картка корпусу",
        "",
        "> Автоматично згенеровано `scrap.py`. Усе, що позначено «перевірте» або TODO, "
        "підтвердіть на сайтах джерел і допишіть вручну.",
        "",
        "## 1. Склад",
        "",
        f"Документів: {len(by_doc)}; фрагментів: {len(all_records)}.",
        "",
        "| doc_id | Назва | Тип | Дата | Мова | Фрагментів | Джерело | Ліцензія |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for doc_id, meta in SOURCES.items():
        lic = meta.get("license") or "TODO"
        how = meta.get("license_how")
        lic_cell = f"{lic} ({how})" if how else lic
        lines.append(f"| {doc_id} | {meta['title']} | {meta['type']} | {meta['date']} | "
                     f"{meta.get('lang', 'uk')} | {by_doc.get(doc_id, 0)} | {meta['url']} | {lic_cell} |")
    lines += ["", "## 2. Правила очищення", ""]
    for name, _, _, desc in TEXT_RULES:
        lines.append(f"- **{name}** — {desc} (спрацювань: {stats.get(name, 0)}).")
    lines += [
        f"- **nav_lines** — рядки навігації сайту (видалено: {stats.get('nav_lines', 0)}).",
        f"- **pdf_header_footer_lines** — для PDF: рядки, що повторюються більш ніж на 30 % сторінок "
        f"(унікальних: {stats.get('pdf_header_footer_lines', 0)}).",
        "- **blank_lines** — понад два порожні рядки поспіль зведено до одного.",
        f"- **duplicate_chunks** — дублікати фрагментів (видалено: {stats.get('duplicate_chunks', 0)}).",
        "- **page_shell** — сторінки, де замість тексту є лише оболонка (без JavaScript).",
        "",
        "Нарізка: по пунктах/абзацах, цільовий розмір "
        f"{TARGET_CHARS} символів (мін. {MIN_CHARS}, макс. {MAX_CHARS}).",
        "",
        "## 3. Sensitivity",
        "",
        "| Рівень | Фрагментів | Обмежений доступ |",
        "|---|---|---|",
    ]
    for level, n in sorted(sens.items()):
        lines.append(f"| {level} | {n} | {'так' if level != 'public' else 'ні'} |")
    lines += [
        "",
        "## 4. Мова",
        "",
        "| Мова | Фрагментів |",
        "|---|---|",
    ]
    for lang, n in sorted(langs.items()):
        lines.append(f"| {lang} | {n} |")
    lines += [
        "",
        "## 5. Відомі прогалини",
        "",
        "- Предметна область: регулювання дорожнього руху та управління транспортними потоками.",
        "",
    ]
    if errors:
        lines += ["## Помилки під час побудови", ""] + [f"- {d}: {e}" for d, e in errors] + [""]
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / "corpus_card.md"
    if path.exists() and not force:
        path = docs / "corpus_card.auto.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    stats, errors, all_records, skipped = Counter(), [], [], []
    for doc_id, meta in SOURCES.items():
        try:
            recs = process_source(doc_id, meta, root, args.refresh, stats)
            all_records += recs
            print(f"[ok]   {doc_id}: {len(recs)} фрагментів")
        except Exception as e:
            if meta.get("optional"):
                print(f"[skip] {doc_id}: недоступне джерело", file=sys.stderr)
                skipped.append(doc_id)
                continue
            errors.append((doc_id, repr(e)))
            print(f"[FAIL] {doc_id}: {e!r}", file=sys.stderr)
            stale = root / "data/clean" / f"{doc_id}.txt"
            if stale.exists():
                stale.unlink()

    for d in skipped:
        SOURCES.pop(d, None)

    out = root / "data/corpus.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = write_card(root, all_records, stats, errors, args.force)
    n_docs = len({r["doc_id"] for r in all_records})
    print(f"\nДокументів: {n_docs}, фрагментів: {len(all_records)} → {out}")
    print(f"Картка корпусу: {card}")


if __name__ == "__main__":
    main()