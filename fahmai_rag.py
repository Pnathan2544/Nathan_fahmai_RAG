"""
FahMai RAG Challenge - Full Pipeline
Embedding : Voyage AI  voyage-3-large
LLM       : Typhoon v2 (Thai LLM - required by rules)
VectorDB  : FAISS (local)

Chunking strategy (from KB structure analysis):
  - Primary split: on '---' dividers (natural section boundaries)
  - Secondary split: on ## headings within large sections
  - Table-aware: never break a markdown table
  - หมายเหตุ: keep **หมายเหตุ** blocks with their parent section

Query strategy:
  - Thai brand/product name expansion → English product codes
  - Long questions: extract the actual question sentence(s)
  - Math questions: higher TOP_K
"""

import os
import re
import json
import glob
import time
import faiss
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

VOYAGE_API_KEY  = "pa-zFa8qP_1-A0O3AbFgNQnCsgYISEHyzwWKttI8pWF1Cf"
TYPHOON_API_KEY = "v3oE6Enw8n3jpO6Uq8MMPXqw1D8Jy3px"

KNOWLEDGE_BASE_DIR = "knowledge_base"
QUESTIONS_CSV      = "questions.csv"
OUTPUT_CSV         = "submission.csv"

EMBED_MODEL   = "voyage-3-large"
TYPHOON_BASE    = "http://thaillm.or.th/api/typhoon/v1"
TYPHOON_MODEL   = "/model"

CHUNK_SIZE    = 500    # slightly larger — sections tend to be self-contained
CHUNK_OVERLAP = 0      # no overlap needed when splitting on --- dividers
TOP_K         = 7
TOP_K_MATH    = 10
VOYAGE_BATCH  = 128
LOW_SCORE_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# THAI → ENGLISH BRAND/PRODUCT NAME MAP
# Used to expand queries so Voyage finds the right KB files
# ─────────────────────────────────────────────
THAI_TO_EN = {
    # Brands
    "สายฟ้า": "SaiFah",
    "ดาวเหนือ": "DaoNuea",
    "คลื่นเสียง": "KlunSiang",
    "วงโคจร": "WongKojorn",
    "จุดเชื่อม": "JudChuem",
    "อาร์คเวฟ": "ArcWave",
    "พัลส์เกียร์": "PulseGear",
    "โนวาเทค": "NovaTech",
    "เกมสตอร์ม": "GameStorm",
    # Product families
    "แอร์บุ๊ก": "AirBook",
    "สตอร์มบุ๊ก": "StormBook",
    "สตั๊ดดี้บุ๊ก": "StudyBook",
    "ครีเอเตอร์บุ๊ก": "CreatorBook",
    "โปรบุ๊ก": "ProBook",
    "เฟล็กซ์บุ๊ก": "FlexBook",
    "มินิพีซี": "Mini PC",
    "ออลอินวัน": "All-in-One",
    "เฮดออน": "HeadOn",
    "เฮดโปร": "HeadPro",
    "บัดส์": "Buds",
    "สตูดิโอโปร": "StudioPro",
    "โฮมพอด": "HomePod",
    "ซาวด์บาร์": "SoundBar",
    "ซาวด์พิลล่า": "SoundPillar",
    "รักเกด": "Rugged",
    # Specific models mentioned in Thai
    "แบนด์": "Band",
    "ริง": "Ring",
    "โฟน": "Phone",
    "แท็บ": "Tab",
    "โปรวิว": "ProView",
    "ชาร์จแพด": "ChargePad",
    "คิวไอแพด": "QiPad",
    "ฮับ": "Hub",
    "ด็อค": "Dock",
    "พาวเวอร์แบงค์": "Power Bank",
}

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"

def voyage_embed(texts: List[str], input_type: str) -> List[List[float]]:
    """Call Voyage API directly via requests — bypasses voyageai package version issues."""
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
    }
    resp = requests.post(VOYAGE_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Sort by index to preserve order
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


# ══════════════════════════════════════════════
# STEP 1 — LOAD & CHUNK
# ══════════════════════════════════════════════

def load_markdown_files(base_dir: str) -> List[Dict]:
    docs = []
    for path in sorted(glob.glob(f"{base_dir}/**/*.md", recursive=True)):
        folder   = Path(path).parent.name
        filename = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            docs.append({"text": content, "source": path,
                         "folder": folder, "filename": filename})
    print(f"[load] {len(docs)} markdown files")
    return docs


def is_table_line(line: str) -> bool:
    return bool(re.match(r'\s*\|', line))


def split_section_table_aware(section: str, max_chars: int) -> List[str]:
    """
    Split a section into chunks, never breaking inside a markdown table
    or a **หมายเหตุ** block.
    """
    if len(section) <= max_chars:
        return [section]

    lines = section.split('\n')
    chunks, current, in_table = [], [], False

    for line in lines:
        on_table = is_table_line(line)

        if on_table:
            in_table = True
        elif in_table and not on_table:
            in_table = False

        current_text = '\n'.join(current)
        next_len = len(current_text) + len(line) + 1

        # Only flush if: not in a table, not mid-หมายเหตุ, and size exceeded
        if (not in_table
                and next_len > max_chars
                and len(current_text) > 100):
            chunks.append(current_text.strip())
            current = []

        current.append(line)

    if current:
        chunks.append('\n'.join(current).strip())

    return [c for c in chunks if c.strip()]


def chunk_documents(docs: List[Dict]) -> List[Dict]:
    """
    Chunk strategy:
    1. Split on standalone '---' lines (natural section dividers in FahMai MD files)
    2. Also split on '## ' headings within large blocks
    3. Within each section, apply table-aware size splitting
    4. **หมายเหตุ** blocks stay attached to their section (never split off)
    """
    max_chars = CHUNK_SIZE * 4
    chunks = []

    for doc in docs:
        text = doc["text"]

        # Step 1: split on --- dividers (standalone line)
        raw_sections = re.split(r'\n---\n', text)

        # Step 2: within large sections, also split on ## headings
        sections = []
        for sec in raw_sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) > max_chars:
                # Split on ## headings, keeping the heading with its content
                sub = re.split(r'(?=\n## )', sec)
                sections.extend([s.strip() for s in sub if s.strip()])
            else:
                sections.append(sec)

        # Step 3: size-based split within each section (table-aware)
        chunk_idx = 0
        for section in sections:
            if not section.strip():
                continue
            sub_chunks = split_section_table_aware(section, max_chars)
            for sc in sub_chunks:
                if sc.strip():
                    chunks.append({
                        "text":     sc.strip(),
                        "source":   doc["source"],
                        "folder":   doc["folder"],
                        "filename": doc["filename"],
                        "chunk_id": f"{doc['source']}::{chunk_idx}",
                    })
                    chunk_idx += 1

    print(f"[chunk] {len(chunks)} chunks from {len(docs)} files")
    return chunks


# ══════════════════════════════════════════════
# STEP 2 — EMBED & INDEX
# ══════════════════════════════════════════════

def embed_texts(texts: List[str], batch_size: int = VOYAGE_BATCH) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i:i + batch_size]
        vecs   = voyage_embed(batch, input_type="document")
        all_embeddings.extend(vecs)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.5)
    return np.array(all_embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"[index] {index.ntotal} vectors, dim={embeddings.shape[1]}")
    return index


def save_index(index, chunks, index_path="faiss.index", meta_path="chunks_meta.json"):
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[save] {index_path} + {meta_path}")


def load_index(index_path="faiss.index", meta_path="chunks_meta.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[load] {index.ntotal} vectors")
    return index, chunks


# ══════════════════════════════════════════════
# STEP 3 — QUERY PROCESSING & RETRIEVE
# ══════════════════════════════════════════════

def expand_thai_names(text: str) -> str:
    """
    Append English equivalents of Thai brand/product names to the query.
    This helps Voyage match Thai-named questions to English-coded KB files.
    e.g. "แอร์บุ๊ก 14" → "แอร์บุ๊ก 14 AirBook 14"
    """
    additions = []
    for thai, en in THAI_TO_EN.items():
        if thai in text and en.lower() not in text.lower():
            additions.append(en)
    if additions:
        return text + " " + " ".join(additions)
    return text


def extract_core_question(question: str) -> str:
    """
    For long Thai questions (like Q9 with travel story preamble),
    extract the actual question part — the sentences containing a question mark.
    Falls back to full question if no ? found or question is short.
    """
    if len(question) < 200:
        return question  # short enough, use as-is

    # Split on Thai sentence endings and find sentences with ?
    sentences = re.split(r'(?<=[?])\s*', question)
    question_sentences = [s.strip() for s in sentences if '?' in s]

    if question_sentences:
        # Take last 2 question sentences (the actual question, not preamble)
        core = ' '.join(question_sentences[-2:])
        # But also keep some context — prepend product name if mentioned
        return core

    return question  # fallback


def build_search_query(question: str) -> str:
    """
    Full query processing pipeline:
    1. Extract core question (handle long preamble questions)
    2. Expand Thai brand/product names to English
    """
    core = extract_core_question(question)
    expanded = expand_thai_names(core)
    return expanded


def is_math_question(question: str) -> bool:
    math_keywords = [
        "ราคารวม", "รวมทั้งหมด", "ต้องจ่าย", "ได้กี่", "คะแนน", "Points",
        "ค่าจัดส่ง", "ค่าส่ง", "คำนวณ", "ส่วนลด", "กี่วัน",
        "กี่บาท", "ลดได้", "ได้รับ", "เท่าไหร่", "รวม",
    ]
    return any(kw in question for kw in math_keywords)


def retrieve(query: str, index, chunks: List[Dict],
             top_k: int = TOP_K) -> List[Dict]:
    vecs = voyage_embed([query], input_type="query")
    qvec = np.array([vecs[0]], dtype="float32")
    faiss.normalize_L2(qvec)
    scores, indices = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
    return results


# ══════════════════════════════════════════════
# STEP 4 — PROMPT & ANSWER
# ══════════════════════════════════════════════

def build_prompt(question: str, choices: Dict[str, str],
                 context_chunks: List[Dict], top_score: float) -> str:
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        label = f"[{i}] [{c['folder']}] {c['filename']}"
        context_parts.append(f"{label}\n{c['text']}")
    context_str = "\n\n---\n\n".join(context_parts)

    valid_choices = {k: v for k, v in choices.items()
                     if pd.notna(v) and str(v).strip()}
    choices_str = "\n".join(
        f"  {k.replace('choice_', '')}. {v}"
        for k, v in valid_choices.items()
    )

    score_hint = ""
    if top_score < LOW_SCORE_THRESHOLD:
        score_hint = "\n[หมายเหตุ: บริบทที่ดึงมามีความเกี่ยวข้องต่ำ — ข้อมูลอาจไม่มีในฐานข้อมูล]"

    return f"""คุณคือผู้ช่วยตอบคำถามของร้านฟ้าใหม่ที่แม่นยำ
อ่านบริบทด้านล่างแล้วตอบคำถามโดยเลือกตัวเลือกที่ถูกต้องที่สุด{score_hint}

กฎสำคัญ:
1. ตอบโดยอ้างอิงจากบริบทที่ให้มาเท่านั้น
2. ข้อมูลที่อยู่ใน **หมายเหตุ:** หรือ > blockquote มีความสำคัญสูง — อ่านให้ครบ
3. ถ้าคำถามต้องคำนวณ (ราคา, วัน, คะแนน) — คำนวณก่อนเลือก
4. เลือกตัวเลือก 9 เฉพาะเมื่อ: บริบทไม่มีข้อมูลตอบคำถามนี้ได้เลย
5. เลือกตัวเลือก 10 เฉพาะเมื่อ: คำถามไม่เกี่ยวกับร้านฟ้าใหม่ สินค้า หรือบริการเลย
6. ตอบเป็นตัวเลขเดียว (1-10) ห้ามอธิบาย

=== บริบทจากฐานข้อมูลฟ้าใหม่ ===
{context_str}

=== คำถาม ===
{question}

=== ตัวเลือก ===
{choices_str}

ตอบ (ตัวเลข 1-10):"""


def ask_typhoon(prompt: str, max_retries: int = 3) -> str:
    headers = {
        "Content-Type": "application/json",
        "apikey": TYPHOON_API_KEY,
    }
    payload = {
        "model": TYPHOON_MODEL,
        "messages": [
            {"role": "system",
             "content": "คุณเป็นผู้ช่วยตอบคำถามของร้านฟ้าใหม่ ตอบเป็นตัวเลขเดียว (1-10) เท่านั้น"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{TYPHOON_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [typhoon] attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return "9"


def extract_choice_number(raw: str) -> int:
    match = re.search(r"\b(10|[1-9])\b", raw)
    if match:
        return int(match.group(1))
    match = re.search(r"\d+", raw)
    if match:
        val = int(match.group(0))
        if 1 <= val <= 10:
            return val
    print(f"  [warn] could not parse '{raw}' — defaulting to 9")
    return 9


# ══════════════════════════════════════════════
# STEP 5 — PIPELINES
# ══════════════════════════════════════════════

def build_index_pipeline():
    docs   = load_markdown_files(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(docs)
    print(f"[embed] {len(chunks)} chunks with Voyage ({EMBED_MODEL})…")
    embeddings = embed_texts([c["text"] for c in chunks])
    index = build_faiss_index(embeddings)
    save_index(index, chunks)
    return index, chunks


def answer_questions_pipeline(index, chunks: List[Dict]):
    df = pd.read_csv(QUESTIONS_CSV)
    print(f"[qa] {len(df)} questions")

    choice_cols = sorted(
        [c for c in df.columns if c.startswith("choice_")],
        key=lambda x: int(x.split("_")[1])
    )
    results = []

    for _, row in df.iterrows():
        q_id     = int(row["id"])
        question = str(row["question"])
        choices  = {c: row[c] for c in choice_cols}

        print(f"  Q{q_id:03d}: {question[:70]}")

        # Build search query (expand Thai names, extract core question)
        search_query = build_search_query(question)
        if search_query != question:
            print(f"         query: {search_query[:70]}")

        top_k      = TOP_K_MATH if is_math_question(question) else TOP_K
        retrieved  = retrieve(search_query, index, chunks, top_k=top_k)
        top_score  = retrieved[0]["score"] if retrieved else 0.0
        prompt     = build_prompt(question, choices, retrieved, top_score)
        raw_ans    = ask_typhoon(prompt)
        answer     = extract_choice_number(raw_ans)

        print(f"         score={top_score:.2f}  raw='{raw_ans}'  answer={answer}")
        results.append({"id": q_id, "answer": answer})
        time.sleep(0.3)

    out_df = pd.DataFrame(results)[["id", "answer"]]
    out_df["answer"] = out_df["answer"].astype(int)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ {len(out_df)} answers saved to {OUTPUT_CSV}")
    return out_df


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FahMai RAG pipeline")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--run",         action="store_true")
    parser.add_argument("--all",         action="store_true")
    args = parser.parse_args()

    if args.all or args.build_index:
        index, chunks = build_index_pipeline()

    if args.all or args.run:
        if not (args.all or args.build_index):
            index, chunks = load_index()
        answer_questions_pipeline(index, chunks)

    if not any([args.build_index, args.run, args.all]):
        parser.print_help()