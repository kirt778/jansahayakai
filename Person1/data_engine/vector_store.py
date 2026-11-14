import os
import re
from pathlib import Path
import ssl
import json
import numpy as np
from typing import List, Dict
from data_engine.matcher import load_all_schemes

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path(__file__).resolve().parent
INDEX_PATH = str(DATA_DIR / "vector_index.npz")
META_PATH = str(DATA_DIR / "vector_meta.json")

_model = None

def _keyword_fallback_retrieve(user_query: str, scheme_ids: List[str] = None, top_k: int = 3) -> List[Dict]:
    schemes = load_all_schemes()
    chunks = []
    for s in schemes:
        sid = s["scheme_id"]
        if scheme_ids and sid not in scheme_ids:
            continue
        base_meta = {
            "scheme_id": sid,
            "scheme_name": s["scheme_name"],
            "source_url": s["source_url"],
            "date_checked": s["date_checked"]
        }
        c1 = f"Scheme: {s['scheme_name']} ({s['ministry']}). State: {s.get('state_specific','All India')}. Benefits: {s['benefits_summary']}. Exclusions: {'; '.join(s.get('exclusions', []))}."
        c2 = f"Scheme: {s['scheme_name']}. Required Documents: {', '.join(s['documents_required'])}. Application Steps: {' '.join(s['application_steps'])}."
        chunks.append({**base_meta, "text_chunk": c1})
        chunks.append({**base_meta, "text_chunk": c2})

    q_words = set(re.findall(r"[a-z0-9]+", user_query.lower()))
    stop = {"what", "how", "can", "i", "apply", "for", "the", "in", "is", "are", "to", "get", "my", "me", "tell", "about", "do", "does", "need", "a", "an", "of", "and", "or"}
    q_clean = q_words - stop
    if not q_clean:
        return []

    out_of_scope_terms = {"visa", "passport", "tourist", "america", "usa", "uk", "canada", "schengen", "bitcoin", "crypto", "stock", "cricket", "ipl", "movie", "weather", "flight", "train", "hotel", "pizza"}
    if q_words & out_of_scope_terms:
        return []

    scored = []
    for ch in chunks:
        c_words = set(re.findall(r"[a-z0-9]+", ch["text_chunk"].lower()))
        overlap = q_clean & c_words
        if overlap:
            score = len(overlap) / max(1.0, len(q_clean) ** 0.5)
            if score >= 0.35:
                scored.append((score, {
                    "text_chunk": ch["text_chunk"],
                    "scheme_name": ch["scheme_name"],
                    "scheme_id": ch["scheme_id"],
                    "source_url": ch["source_url"],
                    "date_checked": ch["date_checked"],
                    "similarity_score": round(float(score), 3)
                }))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]

def retrieve_scheme_context(user_query: str, scheme_ids: List[str] = None, top_k: int = 3) -> List[Dict]:
    return _keyword_fallback_retrieve(user_query, scheme_ids=scheme_ids, top_k=top_k)
