"""Turn a PDF into a list of speakable sentences, each with highlight boxes on the page.

Output shape (cached as JSON):
  pages:     [[width, height], ...]            in PDF points
  sentences: [{"t": text, "r": [[page, x0, y0, x1, y1], ...]}, ...]
  toc:       [[level, title, sentence_index], ...]
"""
import re
import statistics
import unicodedata

import pymupdf

BREAK_HYPHENS = ("‐", "­", "-")  # hyphen at a line end that splits a word
REFERENCE_RE = re.compile(r"^\[\d+\]")
SENT_END_RE = re.compile(r"[.?!][\"'”’)\]]*$")
ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "vs.", "al.", "fig.", "mr.", "mrs.", "dr.", "no.", "approx.", "cf."}
MAX_WORDS = 60


def _body_size(doc):
    """Most common font size in the middle of the book = normal body text."""
    sizes = []
    mid = doc.page_count // 2
    for i in range(max(0, mid - 10), min(doc.page_count, mid + 10)):
        for b in doc[i].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    sizes.extend([round(s["size"], 1)] * len(s["text"]))
    return statistics.mode(sizes) if sizes else 10.0


def _is_mono(font):
    f = font.lower()
    return "mono" in f or "courier" in f or "consol" in f or "code" in f


def _page_words(page, body, page_no):
    """Yield speakable words in reading order: dicts with text, bbox, line key, flags."""
    h = page.rect.height
    blocks = page.get_text("rawdict")["blocks"]
    for bi, b in enumerate(blocks):
        if b["type"] != 0:
            continue
        block_text = "".join(c["c"] for l in b["lines"] for s in l["spans"] for c in s["chars"]).strip()
        if not block_text:
            continue
        # running headers / footers
        if b["bbox"][3] < h * 0.07 or b["bbox"][1] > h * 0.91:
            continue
        # end-of-chapter reference lists: "[12] Author: “Title,” ..."
        if REFERENCE_RE.match(block_text):
            continue
        spans = [s for l in b["lines"] for s in l["spans"]]
        if spans and all(_is_mono(s["font"]) for s in spans if s["chars"] and "".join(c["c"] for c in s["chars"]).strip()):
            continue  # code listing
        first_size = max((s["size"] for s in spans if "".join(c["c"] for c in s["chars"]).strip()), default=body)
        heading = first_size >= body * 1.15
        if heading and block_text.lower() == "references":
            continue
        first_word_of_block = True
        for li, l in enumerate(b["lines"]):
            cur = None
            for s in l["spans"]:
                size = s["size"]
                # small text: footnote markers, footnotes, code listings in small mono
                if (size < body * 0.84 or (_is_mono(s["font"]) and size < body * 0.9)) and not heading:
                    if cur:
                        yield cur
                        cur = None
                    continue
                for c in s["chars"]:
                    ch = c["c"]
                    if ch.isspace():
                        if cur:
                            yield cur
                            cur = None
                        continue
                    x0, y0, x1, y1 = c["bbox"]
                    if cur is None:
                        cur = {"t": ch, "b": [x0, y0, x1, y1], "line": (page_no, bi, li),
                               "heading": heading, "block_start": first_word_of_block}
                        first_word_of_block = False
                    else:
                        cur["t"] += ch
                        bb = cur["b"]
                        bb[0], bb[1], bb[2], bb[3] = min(bb[0], x0), min(bb[1], y0), max(bb[2], x1), max(bb[3], y1)
            if cur:
                cur["line_end"] = True
                yield cur


def _ends_sentence(word, nxt):
    t = word["t"]
    if not SENT_END_RE.search(t):
        return False
    if t.lower() in ABBREVIATIONS:
        return False
    if re.fullmatch(r"[A-Z]\.|\d+\.", t):  # initials like "J." or list numbers like "3."
        return False
    if nxt is None:
        return True
    n = nxt["t"].lstrip("\"'“‘([")
    return not n or n[0].isupper() or n[0].isdigit() or nxt["t"][0] in "•"


def speech_text(text):
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]", "", t)  # citation markers like [10] or [3, 4]
    t = t.replace("‐", "-").replace("•", "").replace("‑", "-")
    for a, b in (("e.g.", "for example"), ("i.e.", "that is"), ("et al.", "et al"),
                 ("vs.", "versus"), ("etc.", "et cetera.")):
        t = t.replace(a, b)
    t = re.sub(r"\b[A-Z]{5,}\b", lambda m: m.group(0).capitalize(), t)  # "CHAPTER" -> "Chapter", keeps acronyms like SQL
    return re.sub(r"\s+", " ", t).strip()


def extract(path):
    doc = pymupdf.open(path)
    body = _body_size(doc)
    pages = [[round(p.rect.width, 1), round(p.rect.height, 1)] for p in doc]

    words = []
    for i, page in enumerate(doc):
        words.extend(_page_words(page, body, i))

    # join words split across lines with a hyphen ("charac‐" + "teristics")
    merged = []
    for w in words:
        if merged:
            prev = merged[-1]
            if (prev.get("line_end") and prev["t"].endswith(BREAK_HYPHENS) and len(prev["t"]) > 1
                    and w["t"][:1].islower() and not w["block_start"]):
                prev.setdefault("parts", [dict(prev, parts=None)])
                prev["parts"].append(w)
                prev["t"] = prev["t"][:-1] + w["t"]
                prev["line_end"] = w.get("line_end")
                continue
        merged.append(dict(w))

    sentences, cur = [], []

    def flush():
        if not cur:
            return
        pieces = []
        for w in cur:
            pieces.extend(w.get("parts") or [w])
        rects = []
        for p in pieces:
            pg = p["line"][0]
            if rects and rects[-1][1] == p["line"]:
                r = rects[-1][0]
                r[1], r[2], r[3], r[4] = min(r[1], p["b"][0]), min(r[2], p["b"][1]), max(r[3], p["b"][2]), max(r[4], p["b"][3])
            else:
                rects.append(([pg] + list(p["b"]), p["line"]))
        text = speech_text(" ".join(w["t"] for w in cur))
        if re.search(r"[A-Za-z0-9]", text):
            sentences.append({"t": text, "r": [[r[0]] + [round(v, 1) for v in r[1:]] for r, _ in rects],
                              "h": 1 if cur[0]["heading"] else 0})
        cur.clear()

    for i, w in enumerate(merged):
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        if cur and (w["heading"] != cur[-1]["heading"] or (w["heading"] and w["block_start"])
                    or w["t"].startswith("•")):
            flush()
        # new paragraph after a finished sentence
        if cur and w["block_start"] and re.search(r"[.?!:\"”)]$", cur[-1]["t"]):
            flush()
        cur.append(w)
        if (not w["heading"] and _ends_sentence(w, nxt)) or (
                len(cur) >= MAX_WORDS and re.search(r"[,;:]$", w["t"])) or len(cur) >= MAX_WORDS + 25:
            flush()
    flush()

    # map table-of-contents entries to the first sentence on or after their page
    first_on_page = {}
    for si, s in enumerate(sentences):
        first_on_page.setdefault(s["r"][0][0], si)
    toc = []
    for level, title, page in doc.get_toc():
        si = next((first_on_page[p] for p in range(page - 1, len(pages)) if p in first_on_page), None)
        if si is not None:
            toc.append([level, title, si])
    return {"pages": pages, "sentences": sentences, "toc": toc}


if __name__ == "__main__":
    import sys, time
    t = time.time()
    out = extract(sys.argv[1])
    print(len(out["sentences"]), "sentences in", round(time.time() - t, 1), "s")
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for s in out["sentences"][lo:lo + 25]:
        print(s["r"][0][0] + 1, "H" if s["h"] else " ", s["t"])
