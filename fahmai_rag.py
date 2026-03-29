# -*- coding: utf-8 -*-
"""
FahMai RAG Challenge — Final Pipeline
======================================
Voyage AI (voyage-3) + KBTG LLM
Section-based chunking + Hybrid retrieval (RRF)
 
Setup:
  pip install voyageai pythainlp rank-bm25 requests python-dotenv numpy
 
Create .env:
  THAILLM_API_KEY=your_key
  VOYAGE_API_KEY=your_key
"""
 
import os
import csv
import re
import time
import json
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
 
# =============================================================================
# CONFIGURATION
# =============================================================================
N_QUESTIONS = 100
DATA_DIR = "."
KB_DIR = "./knowledge_base"
TOP_K = 7
VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 128
RRF_K = 60
CANDIDATE_K = 20  # fetch this many from each retriever before RRF
CONTEXT_BUDGET = 20000  # max chars for retrieved context in prompt
LONG_QUESTION_THRESHOLD = 200  # chars — trigger query extraction
 
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY")


 
if not THAILLM_API_KEY:
    raise ValueError("THAILLM_API_KEY not set")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not set")
 
import voyageai
vo = voyageai.Client(api_key=VOYAGE_API_KEY)
 
 
# =============================================================================
# 1. LLM INTERFACE
# =============================================================================
def ask_llm(messages, model="kbtg", max_retries=5):
    """Call ThaiLLM API with retry."""
    url = f"http://thaillm.or.th/api/{model}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "apikey": THAILLM_API_KEY}
    payload = {
        "model": "/model",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"    Error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    return None
 
 
def parse_answer(text):
    """Extract answer number 1-10 from LLM response."""
    if text is None:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"ANSWER:\s*(\d+)", clean)
    if m:
        return int(m.group(1))
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10:
            return int(d)
    return None
 
 
# =============================================================================
# 2. LOAD DATA
# =============================================================================
def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            choices = {str(i): row[f"choice_{i}"] for i in range(1, 11)}
            questions.append({
                "id": int(row["id"]),
                "question": row["question"],
                "choices": choices,
            })
    return questions
 
 
def load_documents(kb_dir):
    """Load all .md files from knowledge base."""
    kb = Path(kb_dir)
    documents = []
    for fp in sorted(kb.rglob("*.md")):
        text = fp.read_text(encoding="utf-8")
        rel_path = str(fp.relative_to(kb))
        documents.append({"path": rel_path, "text": text, "filename": fp.name})
    return documents
 
 
# =============================================================================
# 3. SECTION-BASED CHUNKING
# =============================================================================
def extract_product_metadata(text):
    """Extract metadata from the top of a product .md file."""
    lines = text.split("\n")
    meta = {}
 
    for line in lines[:12]:
        line = line.strip()
        if line.startswith("# "):
            meta["name"] = line[2:].strip()
        elif line.startswith("รหัสสินค้า:"):
            meta["sku"] = line.split(":", 1)[1].strip()
        elif line.startswith("แบรนด์:"):
            meta["brand"] = line.split(":", 1)[1].strip()
        elif line.startswith("ราคา:"):
            meta["price"] = line.split(":", 1)[1].strip()
        elif line.startswith("สถานะ:"):
            meta["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("หมวดหมู่:"):
            meta["category"] = line.split(":", 1)[1].strip()
 
    return meta
 
 
def build_metadata_prefix(meta):
    """Build a metadata prefix string for a product chunk."""
    parts = []
    if "name" in meta:
        parts.append(f"สินค้า: {meta['name']}")
    if "sku" in meta:
        parts.append(f"SKU: {meta['sku']}")
    if "price" in meta:
        parts.append(f"ราคา: {meta['price']}")
    if "status" in meta:
        parts.append(f"สถานะ: {meta['status']}")
    if "brand" in meta:
        parts.append(f"แบรนด์: {meta['brand']}")
    if "category" in meta:
        parts.append(f"หมวด: {meta['category']}")
    return "[" + " | ".join(parts) + "]\n"
 
 
def split_by_sections(text):
    """Split markdown text by ## headers. Returns list of (header, content) tuples."""
    sections = []
    current_header = ""
    current_lines = []
 
    for line in text.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)
 
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))
 
    return sections
 
 
def chunk_product(doc):
    """Chunk a product document by ## sections with metadata prefix."""
    text = doc["text"]
    meta = extract_product_metadata(text)
    prefix = build_metadata_prefix(meta)
 
    sections = split_by_sections(text)
    chunks = []
 
    for header, content in sections:
        if not content.strip():
            continue
        # Skip the metadata block (before first ##)
        if not header and content.startswith("# "):
            continue
 
        enriched = prefix + content
        chunks.append({
            "text": enriched,          # for dense retrieval (with metadata)
            "raw_text": content,       # for BM25 (without metadata prefix)
            "source": doc["path"],
            "section": header,
            "meta": meta,
        })
 
    return chunks
 
 
def chunk_general_faq(doc):
    """Special chunking for general_faq.md — split each Q/A pair."""
    text = doc["text"]
    chunks = []
    current_section = ""
 
    # Split by --- separators
    parts = re.split(r"\n---\n", text)
 
    for part in parts:
        part = part.strip()
        if not part:
            continue
 
        # Track current ## section
        section_match = re.search(r"^(## .+)$", part, re.MULTILINE)
        if section_match:
            current_section = section_match.group(1)
 
        # Only keep parts that have Q/A content
        if "**Q:" in part or "Q:" in part:
            prefix = f"[FAQ ฟ้าใหม่ | {current_section}]\n"
            chunks.append({
                "text": prefix + part,
                "raw_text": part,
                "source": doc["path"],
                "section": current_section,
                "meta": {"name": "FAQ ฟ้าใหม่"},
            })
        elif part.startswith("## "):
            # Section header without Q/A — might have intro text
            if len(part) > 50:  # only if there's meaningful content
                prefix = f"[FAQ ฟ้าใหม่ | {current_section}]\n"
                chunks.append({
                    "text": prefix + part,
                    "raw_text": part,
                    "source": doc["path"],
                    "section": current_section,
                    "meta": {"name": "FAQ ฟ้าใหม่"},
                })
 
    return chunks
 
 
def chunk_policy_or_storeinfo(doc, label):
    """Chunk policy or store info docs by ## sections."""
    text = doc["text"]
    sections = split_by_sections(text)
    chunks = []
 
    for header, content in sections:
        if not content.strip():
            continue
        if not header and content.startswith("# "):
            # Extract doc title for prefix
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            label = title_match.group(1) if title_match else label
            continue
 
        prefix = f"[{label}]\n"
        chunks.append({
            "text": prefix + content,
            "raw_text": content,
            "source": doc["path"],
            "section": header,
            "meta": {"name": label},
        })
 
    return chunks
 
 
def build_all_chunks(documents):
    """Route each document to the right chunking strategy."""
    all_chunks = []
 
    for doc in documents:
        path = doc["path"]
 
        if path.startswith("products/"):
            all_chunks.extend(chunk_product(doc))
 
        elif path == "store_info/general_faq.md":
            all_chunks.extend(chunk_general_faq(doc))
 
        elif path.startswith("store_info/"):
            name = Path(path).stem.replace("_", " ")
            all_chunks.extend(chunk_policy_or_storeinfo(doc, name))
 
        elif path.startswith("policies/"):
            name = Path(path).stem.replace("_", " ")
            all_chunks.extend(chunk_policy_or_storeinfo(doc, name))
 
        else:
            # Fallback: whole doc as one chunk
            all_chunks.append({
                "text": doc["text"],
                "raw_text": doc["text"],
                "source": path,
                "section": "",
                "meta": {},
            })
 
    return all_chunks
 
 
# =============================================================================
# 4. EMBEDDING — Voyage AI
# =============================================================================
def voyage_embed_batch(texts, input_type="document"):
    """Embed texts using Voyage AI with batching."""
    all_embeddings = []
    for i in range(0, len(texts), VOYAGE_BATCH_SIZE):
        batch = texts[i : i + VOYAGE_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type=input_type)
        all_embeddings.extend(result.embeddings)
        if i + VOYAGE_BATCH_SIZE < len(texts):
            time.sleep(0.3)
    return np.array(all_embeddings, dtype=np.float32)
 
 
def normalize(embeddings):
    """L2-normalize for cosine similarity via dot product."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    return embeddings / norms
 
 
# =============================================================================
# 5. RETRIEVAL
# =============================================================================
def dense_retrieve(query, chunk_embs, k=CANDIDATE_K):
    """Top-k by cosine similarity using Voyage embeddings."""
    q_emb = voyage_embed_batch([query], input_type="query")
    q_emb = normalize(q_emb)
    scores = np.dot(chunk_embs, q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, scores[top_idx]
 
 
def bm25_retrieve(query, bm25_index, k=CANDIDATE_K):
    """Top-k by BM25 keyword matching (case-insensitive)."""
    from pythainlp.tokenize import word_tokenize
    tokens = word_tokenize(query.lower(), engine="newmm")
    scores = bm25_index.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, scores[top_idx]
 
 
def hybrid_retrieve(query, chunk_embs, bm25_index, k=TOP_K):
    """Combine dense + BM25 via Reciprocal Rank Fusion."""
    d_idx, _ = dense_retrieve(query, chunk_embs, k=CANDIDATE_K)
    b_idx, _ = bm25_retrieve(query, bm25_index, k=CANDIDATE_K)
 
    rrf_scores = {}
    for rank, idx in enumerate(d_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(b_idx, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank)
 
    sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
    return sorted_idx
 
 
# =============================================================================
# 6. QUERY PRE-PROCESSING — extract core question from long text
# =============================================================================
EXTRACT_PROMPT = (
    "อ่านข้อความลูกค้าด้านล่าง แล้วสรุปเฉพาะคำถามหลักที่เกี่ยวกับสินค้าหรือบริการของร้านฟ้าใหม่\n\n"
    "กฎ:\n"
    "- ตัดเนื้อหาที่ไม่เกี่ยวข้องออก (เรื่องเล่าส่วนตัว สถานที่ท่องเที่ยว อาหาร)\n"
    "- เก็บชื่อสินค้า รุ่น สเปค และเงื่อนไขสำคัญไว้\n"
    "- เก็บทุกคำถามย่อย (อาจมีมากกว่า 1 คำถาม)\n"
    "- ตอบเป็นคำถามสั้นๆ 1-3 ประโยค ภาษาไทย"
)
 
 
def extract_core_question(question, model="kbtg"):
    """Use LLM to extract the core question from a long message."""
    raw = ask_llm([
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": question},
    ], model=model)
    if raw and len(raw) > 10:
        return raw
    return question  # fallback to original
 
 
# =============================================================================
# 7. PROMPT ENGINEERING
# =============================================================================
SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญของร้านฟ้าใหม่ (FahMai Electronics) ทำหน้าที่ตอบคำถามลูกค้าจากข้อมูลอ้างอิงที่ให้มาเท่านั้น
 
═══ วิธีเลือกคำตอบ ═══
 
ขั้นที่ 1: คำถามเกี่ยวกับฟ้าใหม่หรือไม่?
- ถ้าคำถามไม่เกี่ยวกับสินค้า บริการ หรือร้านฟ้าใหม่เลย → ตอบ ANSWER: 10 ทันที
- ตัวอย่างที่ต้องตอบ 10: สูตรอาหาร, ตั๋วเครื่องบิน, อัตราดอกเบี้ย, วันหยุดราชการ, รายได้ของบริษัท, ราคาหุ้น, สภาพอากาศ, ข่าวการเมือง
- ⚠️ ระวัง: บางคำถามเล่าเรื่องยาวเรื่องส่วนตัว (เที่ยว ออกกำลังกาย ฯลฯ) แต่จบด้วยคำถามเกี่ยวกับสินค้าฟ้าใหม่ → นั่นเกี่ยวข้อง ไม่ใช่ 10
 
ขั้นที่ 2: ข้อมูลที่ต้องการมีอยู่ในข้อมูลอ้างอิงหรือไม่?
- ถ้าคำถามเกี่ยวกับฟ้าใหม่ แต่ข้อมูลเฉพาะที่ถามไม่ปรากฏในข้อมูลอ้างอิง → ตอบ ANSWER: 9
- ตัวอย่างที่ตอบ 9: ค่าซ่อม, คะแนนรีวิว, ค่า SAR, ประเทศที่ผลิต, สินค้ารุ่นที่ไม่มีในระบบ
 
ขั้นที่ 3: เลือกตัวเลือกที่ตรงกับข้อมูลมากที่สุด
- อ่านข้อมูลอ้างอิงให้ละเอียด แล้วเลือกตัวเลือก 1-8 ที่ตรงกับข้อเท็จจริงมากที่สุด
 
═══ คำถามพื้นฐาน ═══
1. ราคาสินค้า สถานะสินค้า (มี/หมด) และหมวดหมู่ สามารถดูได้จาก section แรก #
# Product Name 
 
รหัสสินค้า: SKU
แบรนด์: Brand — แบรนด์ในเครือฟ้าใหม่ / แบรนด์พันธมิตร
หมวดหมู่: Category
ราคา: ฿XX,XXX
สถานะ: มีสินค้า / สินค้าลดล้างสต็อก (CLEARANCE) / Pre-order
วันที่อัปเดต: 1 มีนาคม 2569
 
2. ประกันศูนย์ของสินค้าชิ้นนั้นๆสามารถดูได้จาก section "การรับประกัน" หรือ "เงื่อนไขการรับประกัน"
3. การเปรียบเทียบสินค้ารุ่นต่างๆ ดูได้จาก section ##สเปคสินค้า ที่เป็นตาราง
4. โปรดอ่าน ** ** หมายเหตุพิเศษ** ใน section การรับประกันให้ละเอียด เพราะอาจมีเงื่อนไขที่ไม่ปกติ 
═══ กฎสำคัญ ═══
 
1. ห้ามเดา ห้ามใช้ความรู้นอกเหนือจากข้อมูลอ้างอิง
2. ถ้าข้อมูลอ้างอิงไม่ได้ระบุข้อมูลที่ถาม แม้จะเกี่ยวกับสินค้าฟ้าใหม่ → ตอบ 9
3. เปรียบเทียบทุกตัวเลือกกับข้อมูลอ้างอิงทีละข้อ ระวังตัวเลือกที่ดูคล้ายกันแต่ต่างในรายละเอียดย่อย เช่น eSIM vs nanoSIM, DDR4 vs DDR5, มี OIS vs ไม่มี OIS, มี LDAC vs ไม่มี LDAC
4. ถ้าสินค้ามีหลายรุ่นย่อย (เช่น Tab A5 มีทั้ง LTE และ WiFi) ต้องตอบครบทุกรุ่น
 
═══ ข้อควรระวังเฉพาะร้านฟ้าใหม่ ═══
 
สินค้าและประกัน:
- สินค้า CLEARANCE (ลดล้างสต็อก): ตรวจสอบหมายเหตุพิเศษ อาจมีเงื่อนไข "ไม่รับคืน"
- แบรนด์พันธมิตร (NovaTech, ArcWave, PulseGear, ZenByte) มีประกันจากแบรนด์นั้น ไม่ใช่จากฟ้าใหม่
- NovaTech ไม่มีบริการ On-site ต้องนำส่งศูนย์ NovaTech
- ดาวเหนือแล็ปท็อป: On-site ปีแรก, drop-off ปีที่สอง
- ดาวเหนือเดสก์ท็อป/All-in-One: On-site ตลอด 3 ปี
- Care+ ใช้ได้เฉพาะแบรนด์ฟ้าใหม่ที่ราคาเกิน ฿5,000 ไม่รวมแบรนด์พันธมิตร
- IP68 กันน้ำ แต่ความเสียหายจากน้ำไม่อยู่ในประกัน
 
นโยบายร้าน:
- คืนสินค้าภายใน 15 วัน (Mega Sale: 7 วัน)
- หูฟัง In-ear/TWS ที่เปิดใช้แล้ว ไม่รับคืน (สุขอนามัย)
- สินค้า Bundle ต้องคืนทั้งชุด
- สถานะ "จัดส่งแล้ว" ไม่สามารถยกเลิกได้ ต้องรับแล้วคืน
- Pre-order: ยกเลิก≥3วันก่อนส่ง=ฟรี, <3วัน=หัก5%
 
การคำนวณ:
- Points: ปัดเศษทศนิยมทิ้งก่อนคูณ multiplier แล้วปัดทิ้งอีกครั้ง
  Silver=1pt/฿100, Gold=1.5pt/฿100, Platinum=2pt/฿100
  ตัวอย่าง Gold ซื้อ ฿32,990: floor(32990÷100)=329 → 329×1.5=493.5 → floor=493 Points
- แลก Points: 100pt=฿50, ใช้ได้ไม่เกิน 20% ของราคาสินค้า
- Platinum: Express ฟรีในกรุงเทพฯ+ปริมณฑล
- สินค้าหนัก>30กก.: +฿200, ขนขึ้นชั้น4+: +฿100/ชั้น (ไม่มีลิฟต์)
- รับเป็นเงิน fiat เท่านั้น ไม่รับคริปโต
 
═══ รูปแบบคำตอบ ═══
ถ้าต้องคำนวณ (ราคารวม, คะแนนสะสม, ค่าจัดส่ง): แสดงขั้นตอนสั้นๆ ก่อนตอบ
จบด้วย ANSWER: X (X คือตัวเลข 1-10) เสมอ"""
 
 
def build_rag_prompt(question, choices, retrieved_chunks):
    """Build user prompt with retrieved context."""
    context_parts = []
    total_chars = 0
    for i, c in enumerate(retrieved_chunks):
        source = c["source"]
        section = c.get("section", "")
        section_label = section.replace("## ", "") if section else ""
        label = source
        if section_label:
            label += f" > {section_label}"
        raw = c['raw_text']
        if total_chars + len(raw) > CONTEXT_BUDGET:
            remaining = CONTEXT_BUDGET - total_chars
            if remaining > 500:
                raw = raw[:remaining]
            else:
                break
        total_chars += len(raw)
        context_parts.append(f"[เอกสาร {i+1}: {label}]\n{raw}")
 
    context = "\n\n".join(context_parts)
    choices_text = "\n".join(f"{k}. {v}" for k, v in choices.items())
 
    return (
        f"ข้อมูลอ้างอิง:\n{context}\n\n"
        f"---\n"
        f"คำถาม: {question}\n\n"
        f"ตัวเลือก:\n{choices_text}\n\n"
        f"จากข้อมูลอ้างอิงด้านบน เลือกตัวเลือกที่ถูกต้องที่สุดเพียงข้อเดียว ตอบ ANSWER: X"
    )
 
 
# =============================================================================
# 8. PIPELINE
# =============================================================================
def run_pipeline(questions, chunks, chunk_embs, bm25_index, model="kbtg",
                 n=N_QUESTIONS, extract_model="kbtg", debug=True):
    """Run the full RAG pipeline on n questions."""
    predictions = {}
    errors = 0
    start_time = time.time()
 
    for i, q in enumerate(questions[:n]):
        question = q["question"]
 
        # Query pre-processing for long questions
        retrieval_query = question
        if len(question) > LONG_QUESTION_THRESHOLD:
            print(f"    [Q{q['id']}] Long question ({len(question)} chars), extracting core...")
            retrieval_query = extract_core_question(question, model=extract_model)
            print(f"    → Core: {retrieval_query[:80]}...")
 
        # Retrieve
        top_idx = hybrid_retrieve(retrieval_query, chunk_embs, bm25_index, k=TOP_K)
        retrieved = [chunks[j] for j in top_idx]
 
        # Debug: show retrieved sources
        if debug:
            sources = [f"{c['source'].split('/')[-1]}>{c.get('section','')[:20]}" for c in retrieved]
            print(f"    Retrieved: {sources}")
 
        # Generate (use FULL original question, not extracted one)
        prompt = build_rag_prompt(question, q["choices"], retrieved)
        raw = ask_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], model=model)
 
        pred = parse_answer(raw)
        if pred is None:
            errors += 1
            pred = 1
        predictions[q["id"]] = pred
 
        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        remaining = avg * (n - i - 1)
        status = "⚠️" if pred in (9, 10) else "✓"
        print(f"  [{i+1:>3}/{n}] Q{q['id']:>3}: pred={pred} {status}  (~{remaining:.0f}s left)")
        if debug and raw and pred in (9, 10):
            print(f"    LLM raw: {raw[:100]}")
        time.sleep(0.3)
 
    elapsed = time.time() - start_time
    print(f"\n  {model}: {len(predictions)} answers in {elapsed:.1f}s ({errors} parse errors)")
    return predictions
 
 
# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("FahMai RAG Challenge — Pipeline")
    print("=" * 60)
 
    # --- Load ---
    print("\n[1/5] Loading data...")
    questions = load_questions(f"{DATA_DIR}/questions.csv")
    documents = load_documents(KB_DIR)
    print(f"  {len(questions)} questions, {len(documents)} documents")
 
    # --- Chunk ---
    print("\n[2/5] Chunking documents...")
    chunks = build_all_chunks(documents)
    print(f"  {len(chunks)} chunks created")
 
    # Show distribution
    product_chunks = sum(1 for c in chunks if c["source"].startswith("products/"))
    policy_chunks = sum(1 for c in chunks if c["source"].startswith("policies/"))
    store_chunks = sum(1 for c in chunks if c["source"].startswith("store_info/"))
    print(f"    products: {product_chunks}, policies: {policy_chunks}, store_info: {store_chunks}")
 
    # --- Embed ---
    print(f"\n[3/5] Embedding {len(chunks)} chunks with {VOYAGE_MODEL}...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = voyage_embed_batch(chunk_texts, input_type="document")
    chunk_embeddings = normalize(chunk_embeddings)
    print(f"  Shape: {chunk_embeddings.shape}")
 
    # --- BM25 index ---
    print("\n[4/5] Building BM25 index (case-insensitive)...")
    from pythainlp.tokenize import word_tokenize
    from rank_bm25 import BM25Okapi
 
    tokenized = [word_tokenize(c["raw_text"].lower(), engine="newmm") for c in chunks]
    bm25_index = BM25Okapi(tokenized)
    print(f"  BM25 index: {len(tokenized)} entries")
 
    # --- Run KBTG ---
    print("\n[5/5] Running pipeline with KBTG...")
    print("=" * 60)
    kbtg_preds = run_pipeline(
        questions, chunks, chunk_embeddings, bm25_index,
        model="kbtg", n=N_QUESTIONS, extract_model="kbtg"
    )
 
    # --- Write submission ---
    with open("submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for q in questions:
            writer.writerow([q["id"], kbtg_preds.get(q["id"], 1)])
    print(f"\nsubmission.csv written ({len(questions)} rows)")
 
 
if __name__ == "__main__":
    main()