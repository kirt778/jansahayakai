import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PERSON1 = ROOT / "Person1"
if str(PERSON1) not in sys.path:
    sys.path.insert(0, str(PERSON1))

from data_engine.matcher import evaluate_schemes, load_all_schemes, get_nearest_csc_guidance
from data_engine.vector_store import retrieve_scheme_context
from shared.schema import UserProfile

def to_person1_profile(data: dict[str, Any]) -> UserProfile:
    clean = {k: v for k, v in data.items() if v is not None}
    return UserProfile(**clean)

def run_eligibility(profile_data: dict[str, Any]) -> list[dict[str, Any]]:
    profile = to_person1_profile(profile_data)
    matches = evaluate_schemes(profile)
    rank = {"eligible": 0, "possibly_eligible": 1, "not_eligible": 2}
    matches.sort(key=lambda m: (rank[m.status], m.scheme_name))
    return [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in matches]

def list_schemes() -> list[dict[str, Any]]:
    return load_all_schemes()

def get_scheme(scheme_id: str):
    return next((s for s in load_all_schemes() if s["scheme_id"] == scheme_id), None)

def get_csc_info(state: str = None, language: str = "en") -> dict[str, Any]:
    info = get_nearest_csc_guidance(state)
    if language and language != "en":
        from app.core.voice import translate_text
        return {
            "center_name": translate_text(info["center_name"], "en", language),
            "locator_url": info["locator_url"],
            "helpline": info["helpline"],
            "instructions": translate_text(info["instructions"], "en", language)
        }
    return info

def search_scheme_rag(query: str, scheme_ids: list[str] = None, top_k: int = 3) -> list[dict[str, Any]]:
    return retrieve_scheme_context(query, scheme_ids=scheme_ids, top_k=top_k)
