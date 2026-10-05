# Resume–Job Description Matching System

## Final V3 Hybrid Semantic + Rule-Based Ranking Engine

This project implements a Resume–Job Description (JD) Matching System
that analyzes a resume against a job description and produces an
interpretable matching result.

The final production system uses a lightweight hybrid approach that
combines:

- pretrained SBERT semantic similarity
- rule-based skill coverage
- fuzzy role/title similarity
- experience compatibility
- a fixed-weight composite score
- a capped skill-match bonus
- a rule-based role-domain bonus

The system is designed primarily as an explainable ranking aid for
resume screening rather than as an automated hiring decision-maker.

---

# 1. Project Objective

The main objective is to match a candidate resume with a job
description and rank the candidate according to the degree of
compatibility between the resume and the job requirements.

The system provides:

- final matching score
- Strong / Moderate / Weak decision
- matched skills
- missing skills
- resume role
- JD role
- experience comparison
- explanation of the matching result

The final engine is designed to combine semantic understanding with
explicit, inspectable rules.

---

# 2. Final Methodology

The final production engine is:

**V3 Hybrid Semantic + Rule-Based Ranking Engine**

The complete processing pipeline is:

Resume + Job Description
↓
Text Extraction
↓
Text Preprocessing
↓
Structured Resume/JD Parsing
↓
Model-Ready Matching Text
↓
Feature / Signal Computation
↓
V2 Fixed-Weight Base Score
↓
V3 Bonus Layer
↓
Final V3 Score
↓
Match Decision + Explanation

---

# 3. Processing Pipeline

## 3.1 Document Input

The system accepts:

- Resume file
- Job description file

Supported document processing includes:

- PDF
- TXT
- DOCX

---

## 3.2 Text Extraction

The system extracts textual content from the uploaded documents.

Main implementation:

```text
src/extract.py
```
