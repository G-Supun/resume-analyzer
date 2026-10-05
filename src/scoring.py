from __future__ import annotations

from typing import Iterable, List
from rapidfuzz import fuzz


def safe_get_section(sections: dict, key: str) -> str:
    value = sections.get(key, "")
    return value if isinstance(value, str) else ""


def build_resume_match_text(obj: dict) -> str:
    sections = obj.get("sections", {})

    parts = [
        safe_get_section(sections, "summary"),
        safe_get_section(sections, "skills"),
        safe_get_section(sections, "experience"),
        safe_get_section(sections, "projects"),
        safe_get_section(sections, "education"),
    ]

    text = "\n".join([p for p in parts if p.strip()]).strip()

    if len(text.split()) < 80:
        text = obj.get("model_text", obj.get("normalized_text", ""))

    return text.strip()


def build_jd_match_text(obj: dict) -> str:
    sections = obj.get("sections", {})
    title = obj.get("final_title_v1", "")

    parts = [
        title,
        safe_get_section(sections, "summary"),
        safe_get_section(sections, "responsibilities"),
        safe_get_section(sections, "requirements"),
        safe_get_section(sections, "skills"),
        safe_get_section(sections, "education"),
    ]

    text = "\n".join([p for p in parts if p.strip()]).strip()

    if len(text.split()) < 60:
        text = obj.get("model_text", obj.get("normalized_text", ""))

    return text.strip()


def _normalize_role_label(role_text: str) -> str:
    if not role_text:
        return ""

    r = role_text.lower().strip()

    recruiter_family = [
        "talent acquisition specialist",
        "recruitment consultant",
        "it recruiter",
        "technical recruiter",
        "hr recruiter",
        "recruiter",
        "talent acquisition",
    ]
    teacher_family = [
        "esl teacher",
        "english teacher",
        "teacher",
        "tutor",
        "lecturer",
        "trainer",
    ]

    for term in recruiter_family:
        if term in r:
            return term

    for term in teacher_family:
        if term in r:
            return term

    if "web developer" in r:
        return "web developer"
    if "frontend developer" in r:
        return "frontend developer"
    if "backend developer" in r:
        return "backend developer"
    if "software engineer" in r:
        return "software engineer"
    if "software developer" in r:
        return "software developer"
    if "data analyst" in r:
        return "data analyst"
    if "data scientist" in r:
        return "data scientist"
    if "marketing intern" in r:
        return "marketing intern"
    if "content creator" in r:
        return "content creator"
    if "finance executive" in r:
        return "finance executive"
    if "intern" in r:
        return "intern"

    return r[:80].strip()


def skill_coverage_score(resume_skills: Iterable[str], jd_skills: Iterable[str]) -> float:
    resume_set = set([s.lower().strip() for s in resume_skills if str(s).strip()])
    jd_set = set([s.lower().strip() for s in jd_skills if str(s).strip()])

    if len(jd_set) == 0:
        return 0.5

    return len(resume_set & jd_set) / len(jd_set)


def title_similarity_score(resume_role: str, jd_role: str) -> float:
    rr = _normalize_role_label(resume_role)
    jr = _normalize_role_label(jd_role)

    if not rr or not jr:
        return 0.5

    recruiter_family = {
        "talent acquisition specialist",
        "recruitment consultant",
        "it recruiter",
        "recruiter",
        "talent acquisition",
    }
    teacher_family = {
        "esl teacher",
        "english teacher",
        "teacher",
        "tutor",
    }

    if rr in recruiter_family and jr in recruiter_family:
        return 0.90

    if rr in teacher_family and jr in teacher_family:
        return 0.90

    return fuzz.token_set_ratio(rr, jr) / 100.0


def experience_match_score(
    resume_years: float,
    resume_trusted: bool,
    jd_years: float,
    jd_trusted: bool
) -> float:
    if (not jd_trusted) or jd_years <= 0:
        return 0.5

    if not resume_trusted:
        return 0.4

    if resume_years >= jd_years:
        return 1.0
    elif resume_years >= 0.75 * jd_years:
        return 0.8
    elif resume_years >= 0.50 * jd_years:
        return 0.6
    elif resume_years > 0:
        return 0.3
    else:
        return 0.0


def final_score_v2_formula(
    semantic_score: float,
    skill_score: float,
    title_score: float,
    experience_score: float
) -> float:
    return (
        0.40 * semantic_score +
        0.30 * skill_score +
        0.20 * title_score +
        0.10 * experience_score
    )


def domain_bonus(jd_role: str, resume_role: str, resume_skills: Iterable[str]) -> float:
    jd_role_l = _normalize_role_label(jd_role)
    resume_role_l = _normalize_role_label(resume_role)
    rs = set([x.lower().strip() for x in resume_skills if str(x).strip()])

    bonus = 0.0

    recruiter_family = {
        "talent acquisition specialist",
        "recruitment consultant",
        "it recruiter",
        "recruiter",
        "talent acquisition",
    }
    recruiter_skills = {
        "it recruitment", "recruitment", "sourcing", "screening",
        "applicant tracking systems", "ats", "onboarding",
        "linkedin", "naukri", "monster", "indeed",
        "contract hiring", "end-to-end recruitment",
        "vendor management", "boolean search",
    }

    if jd_role_l in recruiter_family or "recruiter" in jd_role.lower() or "recruitment" in jd_role.lower():
        if resume_role_l in recruiter_family:
            bonus += 0.08
        if len(rs & recruiter_skills) >= 5:
            bonus += 0.07

    if "web developer" in jd_role_l:
        if "web developer" in resume_role_l or "frontend developer" in resume_role_l:
            bonus += 0.08
        if len(rs & {"html", "css", "javascript", "react", "php", "node.js", "nodejs"}) >= 3:
            bonus += 0.05

    if "marketing" in jd_role_l:
        if len(rs & {"marketing", "sales", "customer service", "communication", "microsoft office"}) >= 3:
            bonus += 0.07
        if "intern" in resume_role_l:
            bonus += 0.03

    if "content creator" in jd_role_l or "social media" in jd_role.lower():
        if len(rs & {"social media", "marketing", "instagram", "facebook", "content writing", "photography", "videography", "graphic design"}) >= 3:
            bonus += 0.08
        if len(rs & {"content creation", "video editing", "canva", "storytelling", "tiktok", "youtube"}) >= 1:
            bonus += 0.05

    return round(bonus, 4)


def build_reason(row: dict, matched_skills: List[str], missing_skills: List[str]) -> str:
    reasons = []

    semantic_score = float(row.get("semantic_score", 0.0))
    title_score = float(row.get("title_score", 0.0))
    experience_score = float(row.get("experience_score", 0.0))

    if semantic_score >= 0.72:
        reasons.append("strong semantic match")
    elif semantic_score >= 0.66:
        reasons.append("good semantic match")

    if title_score >= 0.85:
        reasons.append("very close role title")
    elif title_score >= 0.60:
        reasons.append("related role title")

    if len(matched_skills) >= 3:
        reasons.append(f"matches {len(matched_skills)} JD skills")
    elif len(matched_skills) > 0:
        reasons.append(f"matches {len(matched_skills)} JD skill(s)")

    if experience_score >= 0.8:
        reasons.append("experience level fits well")
    elif experience_score == 0.4:
        reasons.append("experience not clearly verified")

    if not reasons:
        reasons.append("moderate overall fit")

    return "; ".join(reasons)