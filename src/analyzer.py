from __future__ import annotations

from pathlib import Path
from typing import Union

from sentence_transformers import SentenceTransformer

from src.extract import extract_text_from_pdf_or_doc
from src.parse_resume import build_resume_json_from_text
from src.parse_jd import build_jd_json_from_text
from src.scoring import (
    build_resume_match_text,
    build_jd_match_text,
    skill_coverage_score,
    title_similarity_score,
    experience_match_score,
    final_score_v2_formula,
    domain_bonus,
    build_reason,
)


PathLike = Union[str, Path]

# lazy-loaded global model
_sbert_model = None


def get_sbert_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


def score_to_decision(final_score_v3: float) -> str:
    if final_score_v3 >= 0.75:
        return "Strong match"
    elif final_score_v3 >= 0.60:
        return "Moderate match"
    return "Weak match"


def analyze_new_resume_vs_new_jd(
    resume_path: PathLike,
    jd_path: PathLike,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """
    Analyze one new resume against one new job description using the final v3 hybrid engine.
    """

    resume_path = Path(resume_path)
    jd_path = Path(jd_path)

    # 1) Extract text
    resume_raw = extract_text_from_pdf_or_doc(resume_path)
    jd_raw = extract_text_from_pdf_or_doc(jd_path)

    # 2) Build structured JSON profiles
    resume_obj = build_resume_json_from_text(resume_raw, doc_id=resume_path.stem)
    jd_obj = build_jd_json_from_text(jd_raw, doc_id=jd_path.stem)

    # 3) Build model-ready matching text
    resume_text = build_resume_match_text(resume_obj)
    jd_text = build_jd_match_text(jd_obj)

    # 4) Semantic similarity
    model = get_sbert_model(model_name=model_name)
    emb_resume, emb_jd = model.encode(
        [resume_text, jd_text],
        normalize_embeddings=True
    )
    semantic_score = float(emb_resume @ emb_jd)

    # 5) Rule-based signals
    resume_skills = resume_obj.get("extracted_skills_v3", [])
    jd_skills = jd_obj.get("extracted_skills_v3", [])

    matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    skill_score = float(skill_coverage_score(resume_skills, jd_skills))

    resume_role = resume_obj.get("resume_role_v1", "")
    jd_role = jd_obj.get("final_title_v1", "")

    title_score = float(title_similarity_score(resume_role, jd_role))

    resume_years = float(resume_obj.get("years_experience_v3", 0.0))
    resume_trusted = bool(resume_obj.get("years_experience_trusted_v3", False))

    jd_years = float(jd_obj.get("jd_years_experience_v3", 0.0))
    jd_trusted = bool(jd_obj.get("jd_years_experience_trusted_v3", False))

    experience_score = float(
        experience_match_score(
            resume_years=resume_years,
            resume_trusted=resume_trusted,
            jd_years=jd_years,
            jd_trusted=jd_trusted,
        )
    )

    # 6) Composite v2 + v3 scoring
    final_score_v2 = float(
        final_score_v2_formula(
            semantic_score=semantic_score,
            skill_score=skill_score,
            title_score=title_score,
            experience_score=experience_score,
        )
    )

    matched_skill_count = len(matched_skills)
    skill_boost = min(matched_skill_count * 0.01, 0.08)
    dom_bonus = float(domain_bonus(jd_role, resume_role, resume_skills))

    final_score_v3 = float(final_score_v2 + skill_boost + dom_bonus)

    # 7) Decision + explanation
    decision = score_to_decision(final_score_v3)

    reason_row = {
        "semantic_score": semantic_score,
        "title_score": title_score,
        "experience_score": experience_score,
    }
    explanation = build_reason(reason_row, matched_skills, missing_skills)

    return {
        "resume_file": str(resume_path),
        "jd_file": str(jd_path),
        "resume_role": resume_role,
        "jd_role": jd_role,
        "jd_type": jd_obj.get("jd_type_v1", ""),
        "semantic_score": round(semantic_score, 4),
        "skill_score": round(skill_score, 4),
        "title_score": round(title_score, 4),
        "experience_score": round(experience_score, 4),
        "final_score_v2": round(final_score_v2, 4),
        "skill_boost": round(skill_boost, 4),
        "domain_bonus": round(dom_bonus, 4),
        "final_score_v3": round(final_score_v3, 4),
        "decision": decision,
        "matched_skill_count": matched_skill_count,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_years_experience": resume_years,
        "jd_years_experience": jd_years,
        "explanation": explanation,
    }