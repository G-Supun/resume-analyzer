from __future__ import annotations

import re
from typing import Dict, List

from src.preprocess import (
    clean_text,
    normalize_spaces,
    extract_nonempty_lines,
    build_model_ready_text,
)

# ------------------------------------------------------------
# Skill dictionary
# ------------------------------------------------------------
SKILL_PHRASES_V3 = sorted(set([
    # technical
    "python", "java", "sql", "mysql", "power bi", "excel", "html", "css",
    "javascript", "react", "node.js", "nodejs", "php", "c", "c++", "c#",
    "machine learning", "data analysis", "data analytics", "nlp",
    "api", "rest api", "git", "github", "firebase",
    "cloud computing", "security solutions", "cloud technologies", "devops",

    # business / marketing
    "marketing", "sales", "customer service", "project management",
    "communication", "leadership", "microsoft office", "business analysis",
    "presentation", "teamwork",

    # content / social media
    "social media", "content creation", "content writing",
    "facebook", "instagram", "tiktok", "youtube",
    "canva", "meta business suite", "videography", "video editing",
    "photography", "storytelling", "graphic design", "copywriting",

    # recruiter / HR / TA
    "it recruitment",
    "recruitment",
    "recruiting",
    "talent acquisition",
    "technical sourcing",
    "end-to-end recruitment",
    "sourcing",
    "screening",
    "screening calls",
    "boolean search",
    "headhunting",
    "cold calling",
    "mass mailing",
    "stakeholder management",
    "offer negotiation",
    "salary negotiation",
    "candidate engagement",
    "onboarding",
    "interview coordination",
    "vendor management",
    "contract hiring",
    "perm hiring",
    "permanent hiring",
    "contract recruitment",
    "contract-to-hire",
    "c2h",
    "w2",
    "c2c",
    "1099",
    "rate discussions",
    "rate discussion",
    "ats",
    "applicant tracking systems",
    "applicant tracking system",
    "linkedin recruiter",
    "linkedin",
    "naukri",
    "monster",
    "indeed",
    "careerbuilder",
    "compliance",
    "background verification",
    "pipeline management",
    "candidate sourcing",
    "technical screening",

    # teaching / education
    "lesson planning",
    "classroom management",
    "digital pedagogy",
    "teaching",
    "tutoring",
    "assessment",
    "research",
    "data entry",
]), key=len, reverse=True)

# ------------------------------------------------------------
# Role hints
# ------------------------------------------------------------
ROLE_HINTS = sorted(set([
    # recruiter family
    "talent acquisition specialist",
    "recruitment consultant",
    "it recruiter",
    "technical recruiter",
    "hr recruiter",
    "recruiter",
    "talent acquisition",
    "hr executive",

    # teaching family
    "esl teacher",
    "english teacher",
    "teacher",
    "tutor",
    "lecturer",
    "trainer",
    "freelancer",

    # dev / data
    "web developer",
    "software engineer",
    "software developer",
    "frontend developer",
    "backend developer",
    "full stack developer",
    "data analyst",
    "data scientist",
    "machine learning engineer",

    # other common
    "marketing intern",
    "marketing executive",
    "finance executive",
    "content creator",
    "social media content creator",
    "social media executive",
    "intern",
]), key=len, reverse=True)

# ------------------------------------------------------------
# Resume section headers
# ------------------------------------------------------------
RESUME_SECTION_PATTERNS = {
    "summary": r"(summary|professional summary|profile|objective|about me)",
    "skills": r"(skills|top skills|technical skills|key skills|competencies)",
    "experience": r"(experience|work experience|employment|professional experience|internship)",
    "education": r"(education|academic|qualifications)",
    "projects": r"(projects|project experience)",
    "certifications": r"(certifications|certificates)",
    "activities": r"(activities|extracurricular|volunteer)",
    "references": r"(references)",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_role_label(role_text: str) -> str:
    if not role_text:
        return ""

    r = role_text.lower().strip()

    recruiter_terms = {
        "talent acquisition specialist",
        "recruitment consultant",
        "it recruiter",
        "technical recruiter",
        "hr recruiter",
        "recruiter",
        "talent acquisition",
        "hr executive",
    }
    teacher_terms = {
        "esl teacher",
        "english teacher",
        "teacher",
        "tutor",
        "lecturer",
        "trainer",
    }

    for term in recruiter_terms:
        if term in r:
            return term

    for term in teacher_terms:
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

    return sorted(set(found))


def _extract_role_from_text(text: str) -> str:
    lines = extract_nonempty_lines(text)
    low = text.lower()

    # first lines often contain title
    for line in lines[:8]:
        line_l = line.lower()

        if any(x in line_l for x in ["linkedin", "email:", "address:", "phone:", "professional summary"]):
            continue

        if re.search(
            r"(talent acquisition specialist|recruitment consultant|it recruiter|technical recruiter|hr recruiter|recruiter|"
            r"esl teacher|english teacher|teacher|tutor|developer|engineer|analyst|manager|executive|specialist|intern)",
            line_l
        ):
            return _normalize_role_label(line_l)

    # explicit title patterns
    patterns = [
        r"\bposition\s*:\s*([^\n]+)",
        r"\brole\s*:\s*([^\n]+)",
        r"\btitle\s*:\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return _normalize_role_label(m.group(1).strip())

    # fallback hints
    for role in ROLE_HINTS:
        if role in low:
            return _normalize_role_label(role)

    if "intern" in low:
        return "intern"

    return ""


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
    ]

    values: List[float] = []
    for p in patterns:
        for m in re.findall(p, t):
            try:
                val = float(m)
                if 0 <= val <= 40:
                    values.append(val)
            except Exception:
                pass

    if values:
        return max(values), True

    return 0.0, False


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------
def build_resume_json_from_text(text: str, doc_id: str = "NEW_RESUME") -> dict:
    cleaned = clean_text(text)
    cleaned = normalize_spaces(cleaned)

    sections = _split_into_sections(cleaned, RESUME_SECTION_PATTERNS)

    # fallback if section detection fails
    if sum(bool(v.strip()) for v in sections.values()) == 0:
        sections["summary"] = cleaned[:1500]

    extracted_skills = _extract_skills_v3(cleaned)
    resume_role = _extract_role_from_text(cleaned)
    years_exp, years_trusted = _extract_years_experience(cleaned)

    return {
        "doc_id": doc_id,
        "normalized_text": cleaned,
        "model_text": build_model_ready_text(cleaned, max_chars=2500),
        "sections": sections,
        "extracted_skills_v3": extracted_skills,
        "resume_role_v1": resume_role,
        "years_experience_v3": years_exp,
        "years_experience_trusted_v3": years_trusted,
    }