from __future__ import annotations

import re
from typing import Dict, List

from src.preprocess import (
    clean_text,
    normalize_spaces,
    extract_nonempty_lines,
    build_model_ready_text,
)
from src.parse_resume import SKILL_PHRASES_V3


# ------------------------------------------------------------
# JD section headers
# ------------------------------------------------------------
JD_SECTION_PATTERNS = {
    "summary": r"(summary|about the role|overview|job overview|about)",
    "responsibilities": r"(responsibilities|duties|key responsibilities|what you will do)",
    "requirements": r"(requirements|qualifications|eligibility|what we are looking for)",
    "skills": r"(skills|required skills|preferred skills)",
    "education": r"(education|academic qualifications|degree)",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _split_into_sections(text: str, section_patterns: Dict[str, str]) -> Dict[str, str]:
    lines = extract_nonempty_lines(text)
    sections = {k: "" for k in section_patterns.keys()}

    current = None
    bucket: List[str] = []

    def flush():
        nonlocal current, bucket
        if current is not None and bucket:
            joined = "\n".join(bucket).strip()
            if joined:
                sections[current] = (sections[current] + "\n" + joined).strip()
        bucket = []

    for line in lines:
        line_norm = line.lower().strip(" :-")

        matched_key = None
        for sec, pat in section_patterns.items():
            if re.fullmatch(pat, line_norm, flags=re.IGNORECASE):
                matched_key = sec
                break

        if matched_key is not None:
            flush()
            current = matched_key
        else:
            if current is not None:
                bucket.append(line)

    flush()
    return sections


def _extract_skills_v3(text: str) -> List[str]:
    t = " " + text.lower() + " "
    found: List[str] = []

    for skill in SKILL_PHRASES_V3:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, t):
            found.append(skill)

    # helpful JD-specific additions
    if "cloud computing" in t:
        found.append("cloud computing")
    if "security solutions" in t:
        found.append("security solutions")

    # remove noisy recruiter verb hits for non-recruiter docs
    recruiter_context_terms = [
        "talent acquisition", "recruitment consultant", "it recruiter",
        "technical recruiter", "hr recruiter", "recruiter", "recruitment",
        "contract hiring", "onboarding", "candidate sourcing", "ats"
    ]
    has_real_recruiter_context = any(term in t for term in recruiter_context_terms)

    if not has_real_recruiter_context:
        found = [x for x in found if x not in {"recruiting", "recruitment", "end-to-end recruitment"}]

    return sorted(set(found))


def _extract_years_experience(text: str) -> tuple[float, bool]:
    t = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience",
        r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?",
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+experience",
        r"minimum\s+(\d+(?:\.\d+)?)\+?\s+years?",
        r"about\s+(\d+(?:\.\d+)?)\s+years?",
        r"around\s+(\d+(?:\.\d+)?)\s+years?",
        r"over\s+(\d+(?:\.\d+)?)\+?\s+years?",
        r"with\s+(\d+(?:\.\d+)?)\+?\s+years?",
        r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s+yrs?",
    ]

    values: List[float] = []

    for p in patterns[:-1]:
        for m in re.findall(p, t):
            try:
                val = float(m)
                if 0 <= val <= 40:
                    values.append(val)
            except Exception:
                pass

    # range pattern like 2 to 8 yrs
    for m1, m2 in re.findall(patterns[-1], t):
        try:
            v1 = float(m1)
            v2 = float(m2)
            if 0 <= v1 <= 40 and 0 <= v2 <= 40:
                values.append(v1)
        except Exception:
            pass

    if values:
        return max(values), True

    return 0.0, False


def _guess_jd_title(text: str) -> str:
    lines = extract_nonempty_lines(text)
    low = text.lower()

    # explicit Position line
    m = re.search(r"\bposition\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        if 3 <= len(title) <= 100:
            return title

    # skip metadata lines
    skip_prefixes = [
        "company -", "about -", "job type:", "location:", "employment type:",
        "about the role:", "qualifications:", "compensation package -"
    ]
    skip_exact = {
        "job description", "about", "qualifications", "why join us?"
    }

    for line in lines[:15]:
        line_l = line.lower().strip()

        if line_l in skip_exact:
            continue
        if any(line_l.startswith(p) for p in skip_prefixes):
            continue

        if 3 <= len(line) <= 90 and re.search(
            r"(recruiter|consultant|developer|engineer|analyst|manager|teacher|executive|intern|specialist|officer|creator)",
            line_l
        ):
            return line

    # special screening / research-study type
    if ("paid consultancy session" in low or "online research study" in low or "online questionnaire" in low):
        if "cloud" in low or "security" in low:
            return "Cloud / Security Professional Screening Study"
        return "Professional Screening Study"

    # fallback heuristics
    if "recruitment consultant" in low:
        return "Recruitment Consultant"
    if "recruiter" in low and ("it" in low or "technology" in low):
        return "IT Recruiter"
    if "recruiter" in low:
        return "Recruiter"
    if "teacher" in low:
        return "Teacher"
    if "content creator" in low:
        return "Content Creator"
    if "marketing" in low:
        return "Marketing Role"
    if "finance" in low:
        return "Finance Role"

    return lines[0] if lines else "Unknown JD Title"


def _detect_jd_type(text: str) -> str:
    low = text.lower()

    if ("paid consultancy session" in low or "online research study" in low or "online questionnaire" in low):
        return "research_screening"
    if "job type" in low or "employment type" in low or "responsibilities" in low:
        return "standard_job_description"
    return "generic_document"


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------
def build_jd_json_from_text(text: str, doc_id: str = "NEW_JD") -> dict:
    cleaned = clean_text(text)
    cleaned = normalize_spaces(cleaned)

    sections = _split_into_sections(cleaned, JD_SECTION_PATTERNS)

    # fallback if section detection fails
    if sum(bool(v.strip()) for v in sections.values()) == 0:
        sections["summary"] = cleaned[:1500]

    extracted_skills = _extract_skills_v3(cleaned)
    final_title = _guess_jd_title(cleaned)
    jd_years, jd_years_trusted = _extract_years_experience(cleaned)
    jd_type = _detect_jd_type(cleaned)

    return {
        "doc_id": doc_id,
        "normalized_text": cleaned,
        "model_text": build_model_ready_text(cleaned, max_chars=2500),
        "sections": sections,
        "extracted_skills_v3": extracted_skills,
        "final_title_v1": final_title,
        "jd_years_experience_v3": jd_years,
        "jd_years_experience_trusted_v3": jd_years_trusted,
        "jd_type_v1": jd_type,
    }