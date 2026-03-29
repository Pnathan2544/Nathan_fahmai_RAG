#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script: inspect retrieval for failing questions.
Run this AFTER fahmai_rag.py has built chunks and embeddings.

Usage: python debug_retrieval.py
"""

import csv, re, os, sys
import numpy as np
from pathlib import Path

# --- Import chunking from main script ---
# We duplicate the essentials here so this runs standalone
DATA_DIR = "."
KB_DIR = "./knowledge_base"
TOP_K = 7
CANDIDATE_K = 20
RRF_K = 60

# Questions that KBTG said 9 but v5 had a real answer (retrieval failures)
RETRIEVAL_FAILURES = [4, 6, 11, 22, 24, 47, 55, 63, 71, 76, 77, 79, 87, 96, 99]

# Questions where LLM picked wrong answer (reasoning errors)  
LLM_ERRORS = [10, 14, 20, 25, 34, 39, 82, 83, 85, 86, 88, 89, 91, 92, 94, 95, 97, 98, 100]

# v5 ground truth for these questions
V5_ANSWERS = {
    4:6, 6:8, 10:7, 11:4, 14:4, 18:10, 20:2, 22:6, 24:3, 25:5,
    34:5, 39:4, 47:2, 55:10, 63:10, 71:3, 76:5, 77:2, 79:6,
    82:8, 83:5, 85:4, 86:3, 87:7, 88:3, 89:4, 91:2, 92:4,
    94:8, 95:4, 96:5, 97:2, 98:5, 99:4, 100:3
}

def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            choices = {str(i): row[f"choice_{i}"] for i in range(1, 11)}
            questions.append({"id": int(row["id"]), "question": row["question"], "choices": choices})
    return questions

# ============================================================
# Chunking (same as fahmai_rag.py)
# ============================================================
def extract_product_metadata(text):
    lines = text.split("\n")
    meta = {}
    for line in lines[:12]:
        line = line.strip()
        if line.startswith("# "): meta["name"] = line[2:].strip()
        elif line.startswith("รหัสสินค้า:"): meta["sku"] = line.split(":", 1)[1].strip()
        elif line.startswith("แบรนด์:"): meta["brand"] = line.split(":", 1)[1].strip()
        elif line.startswith("ราคา:"): meta["price"] = line.split(":", 1)[1].strip()
        elif line.startswith("สถานะ:"): meta["status"] = line.split(":", 1)[1].strip()
        elif line.startswith("หมวดหมู่:"): meta["category"] = line.split(":", 1)[1].strip()
    return meta

def build_metadata_prefix(meta):
    parts = []
    for k, label in [("name","สินค้า"),("sku","SKU"),("price","ราคา"),("status","สถานะ"),("brand","แบรนด์"),("category","หมวด")]:
        if k in meta: parts.append(f"{label}: {meta[k]}")
    return "[" + " | ".join(parts) + "]\n"

def split_by_sections(text):
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
    text = doc["text"]
    meta = extract_product_metadata(text)
    prefix = build_metadata_prefix(meta)
    sections = split_by_sections(text)
    chunks = []
    for header, content in sections:
        if not content.strip(): continue
        if not header and content.startswith("# "): continue
        chunks.append({
            "text": prefix + content,
            "raw_text": content,
            "source": doc["path"],
            "section": header,
            "meta": meta,
        })
    return chunks

def chunk_general_faq(doc):
    text = doc["text"]
    chunks = []
    current_section = ""
    parts = re.split(r"\n---\n", text)
    for part in parts:
        part = part.strip()
        if not part: continue
        section_match = re.search(r"^(## .+)$", part, re.MULTILINE)
        if section_match: current_section = section_match.group(1)
        if "**Q:" in part or "Q:" in part:
            prefix = f"[FAQ ฟ้าใหม่ | {current_section}]\n"
            chunks.append({
                "text": prefix + part,
                "raw_text": part,
                "source": doc["path"],
                "section": current_section,
                "meta": {"name": "FAQ ฟ้าใหม่"},
            })
        elif part.startswith("## ") and len(part) > 50:
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
    text = doc["text"]
    sections = split_by_sections(text)
    chunks = []
    for header, content in sections:
        if not content.strip(): continue
        if not header and content.startswith("# "):
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

def build_all_chunks(kb_dir):
    kb = Path(kb_dir)
    documents = []
    for fp in sorted(kb.rglob("*.md")):
        text = fp.read_text(encoding="utf-8")
        rel_path = str(fp.relative_to(kb))
        documents.append({"path": rel_path, "text": text, "filename": fp.name})
    
    all_chunks = []
    for doc in documents:
        path = doc["path"]
        if path.startswith("products/") or path.startswith("products\\"):
            all_chunks.extend(chunk_product(doc))
        elif "general_faq" in path:
            all_chunks.extend(chunk_general_faq(doc))
        elif path.startswith("store_info/") or path.startswith("store_info\\"):
            name = Path(path).stem.replace("_", " ")
            all_chunks.extend(chunk_policy_or_storeinfo(doc, name))
        elif path.startswith("policies/") or path.startswith("policies\\"):
            name = Path(path).stem.replace("_", " ")
            all_chunks.extend(chunk_policy_or_storeinfo(doc, name))
    return all_chunks

# ============================================================
# MAIN DEBUG
# ============================================================
def main():
    print("Loading questions...")
    questions = load_questions(f"{DATA_DIR}/questions.csv")
    q_by_id = {q["id"]: q for q in questions}

    print("Building chunks...")
    chunks = build_all_chunks(KB_DIR)
    print(f"  {len(chunks)} chunks total")

    # --- For each failing question, find the CORRECT chunk via brute-force search ---
    print("\n" + "=" * 70)
    print("PART 1: WHERE IS THE CORRECT ANSWER IN THE CHUNKS?")
    print("=" * 70)
    
    # Search terms for each failing question
    search_terms = {
        1:  ["Watch S3 Ultra", "ATM"],
        4:  ["Trade-in", "เทิร์น", "เครื่องเก่า"],
        6:  ["crypto", "Cryptocurrency", "Bitcoin"],
        11: ["CreatorBook 16 OLED", "จอเพิ่ม", "Thunderbolt"],
        22: ["Care+", "จอแตก", "screen"],
        24: ["NovaBuds Pro", "warranty", "ประกัน"],
        47: ["All-in-One 27", "ดาวเหนือ", "27 นิ้ว 4K"],
        55: ["รายได้", "revenue"],
        63: ["กระเพรา", "ผัด", "สูตร"],
        71: ["HeadPro X1", "X1 SE", "ต่าง"],
        76: ["StormBook G5", "DDR4", "DDR5", "RAM"],
        77: ["StormBook G7", "Mini PC M1", "on-site"],
        79: ["SlimBook 14", "AirBook 14", "ประกัน", "on-site"],
        87: ["3,500", "หูฟัง", "TWS", "ครอบหู"],
        96: ["ChargePad 15W", "NovaBuds Pro", "ชาร์จไร้สาย", "Qi"],
        99: ["Power Bank 30,000", "เกาะ", "สมุย", "จัดส่ง"],
    }

    for qid in RETRIEVAL_FAILURES:
        q = q_by_id.get(qid)
        if not q: continue
        
        terms = search_terms.get(qid, [])
        v5_answer = V5_ANSWERS.get(qid, "?")
        
        print(f"\n--- Q{qid} (v5 answer: {v5_answer}) ---")
        print(f"Question: {q['question'][:100]}...")
        print(f"Correct choice {v5_answer}: {q['choices'][str(v5_answer)][:80]}...")
        
        # Search all chunks for matching terms
        matching_chunks = []
        for i, c in enumerate(chunks):
            text_lower = c["raw_text"].lower()
            matches = [t for t in terms if t.lower() in text_lower]
            if len(matches) >= 1:
                matching_chunks.append((i, c, matches))
        
        if matching_chunks:
            print(f"  Found {len(matching_chunks)} chunks with search terms:")
            for idx, c, matches in matching_chunks[:5]:
                section = c["section"].replace("## ", "") if c["section"] else "(top)"
                print(f"    chunk[{idx}] {c['source']} > {section}")
                print(f"      matched: {matches}")
                print(f"      text: {c['raw_text'][:120]}...")
        else:
            print(f"  ⚠️ NO CHUNKS FOUND with terms {terms}")
            print(f"  → This means the answer might not exist OR search terms are wrong")

    # --- PART 2: Check section distribution ---
    print("\n" + "=" * 70)
    print("PART 2: SECTION DISTRIBUTION IN CHUNKS")
    print("=" * 70)
    from collections import Counter
    sections = Counter()
    for c in chunks:
        s = c["section"] if c["section"] else "(preamble)"
        sections[s] += 1
    
    for s, count in sections.most_common(20):
        print(f"  {count:>4}x  {s}")

    # --- PART 3: Sample chunks for Q1 (Watch S3 Ultra) ---
    print("\n" + "=" * 70)
    print("PART 3: ALL CHUNKS FOR Watch S3 Ultra")
    print("=" * 70)
    for i, c in enumerate(chunks):
        if "watch_s3_ultra" in c["source"].lower():
            section = c["section"].replace("## ", "") if c["section"] else "(top)"
            print(f"  chunk[{i}] section='{section}' ({len(c['raw_text'])} chars)")
            print(f"    text[:150]: {c['raw_text'][:150]}...")
            # Check if ATM is in this chunk
            if "ATM" in c["raw_text"] or "atm" in c["raw_text"].lower():
                print(f"    ★★★ CONTAINS ATM ★★★")
            print()

if __name__ == "__main__":
    main()
