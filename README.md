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

The extracted text is passed to the preprocessing and parsing stages.

## 3.3 Text Preprocessing

The preprocessing stage performs operations such as:

- text cleaning
- whitespace normalization
- section preparation
- model-ready text construction

Main implementation:
`src/preprocess.py`

## 3.4 Structured Parsing

The extracted documents are converted into structured representations.

### Resume Parsing

Main implementation:
`src/parse_resume.py`

The resume parser extracts information such as:

- resume role
- skills
- experience
- education
- relevant sections
- years of experience

### Job Description Parsing

Main implementation:
`src/parse_jd.py`

The JD parser extracts information such as:

- job title
- required skills
- responsibilities
- education
- years of experience
- job type / role information

## 4. Semantic Matching

The semantic matching component uses: `sentence-transformers/all-MiniLM-L6-v2`

The model is a pretrained Sentence-BERT (SBERT) model. No task-specific fine-tuning is performed. The resume and JD matching texts are converted into normalized embeddings.

Semantic similarity is calculated as the dot product of the normalized embeddings, which is equivalent to cosine similarity.

## 5. Rule-Based Matching Signals

The semantic score is combined with explicit rule-based signals. The main signals are:

- Skill coverage
- Role/title similarity
- Experience compatibility

These signals provide additional information that is not represented only by the semantic embedding score.

### 5.1 Skill Coverage

The skill score measures the proportion of JD skills that are also identified in the resume.

Conceptually:

```text
skill_score = matched JD skills / total JD skills
```

A fixed skill vocabulary is used by the current implementation. If no JD skills are identified, a neutral default is used. Because the vocabulary is fixed, unseen skills, synonyms, abbreviations, and domain-specific terminology may not always be captured by this signal.

### 5.2 Role / Title Similarity

Role titles are normalized into recognised role labels and compared.

The implementation uses: `RapidFuzz` token-set similarity.

Predefined role families can receive a high similarity when they represent closely related occupations. Examples include:

- recruiter
- talent acquisition
- teacher
- web developer
- software developer
- data analyst
- content creator

### 5.3 Experience Matching

Resume and JD experience requirements are compared using deterministic rules.
The experience score uses tiers depending on how the candidate's experience compares with the JD requirement. Trusted and untrusted extracted experience values are handled using conservative defaults.

## 6. V2 Fixed-Weight Base Score

The four main signals are combined using a fixed-weight formula.

```text
final_score_v2 =
    0.40 × semantic_score
  + 0.30 × skill_score
  + 0.20 × title_score
  + 0.10 × experience_score
```

The semantic signal receives the largest weight, followed by skill coverage, title similarity, and experience compatibility. The weights are fixed in the final implementation.

## 7. V3 Bonus Layer

The V3 engine extends the V2 base score using two explicit bonuses:

```text
final_score_v3 = final_score_v2 + skill_boost + domain_bonus
```

### 7.1 Skill-Match Boost

The skill-match boost is based on the number of matched JD skills.

```text
skill_boost = min(matched_skill_count × 0.01, 0.08)
```

Therefore:

- 1 matched skill → +0.01
- 2 matched skills → +0.02
- ...
- 8 matched skills → +0.08

The boost is capped at: `+0.08`

### 7.2 Role-Domain Bonus

The role-domain bonus provides an additional rule-based adjustment when a resume and JD share a recognised occupational/job-family context.

In this project, "domain" refers to the occupational/job-family domain rather than the employer's industry. The current implementation contains explicit rules for selected role families including:

- Recruiter / Recruitment
- Web Developer
- Marketing
- Content Creator / Social Media

The rules use role and skill evidence. The bonus is deterministic and inspectable.

## 8. Match Decision

The final V3 score is mapped to an application-level decision.

| Final V3 Score | Decision       |
| :------------- | :------------- |
| >= 0.75        | Strong match   |
| 0.60 – < 0.75  | Moderate match |
| < 0.60         | Weak match     |

The score is a matching score and should not be interpreted as a classification accuracy.

## 9. Explainable Output

For each resume-JD pair, the system can provide:

- Resume Role
- JD Role
- Semantic Score
- Skill Score
- Title Score
- Experience Score
- V2 Score
- Skill Boost
- Domain Bonus
- Final V3 Score
- Decision
- Matched Skills
- Missing Skills
- Explanation

The explanation is generated from the matching signals. For example, the system may identify:

- strong semantic match
- very close role title
- multiple matched JD skills
- suitable experience level

This allows users to understand why a candidate received a particular score.

## 10. Research Dataset

The research corpus contains:

- 62 resumes
- 40 job descriptions

The resumes were obtained from actual job applications with applicant consent. Personally identifying information was removed where applicable before processing. The job descriptions were obtained from publicly available job-posting websites.

The documents were initially processed and evaluated using a Google Colab research notebook. The final application implementation was subsequently organized as a modular Python project.

## 11. Pair Generation

Every resume was paired with every job description.
Therefore: `62 × 40 = 2,480 resume-JD pairs`

All 2,480 pairs were automatically scored by the engine. The complete research dataset and generated outputs are not included in this repository because they contain private or generated research data.

## 12. Benchmark Evaluation

A 24-pair benchmark was created from the automatically scored pairs.

| Label     | Count  |
| :-------- | :----- |
| Strong    | 6      |
| Partial   | 9      |
| Poor      | 9      |
| **Total** | **24** |

The benchmark labels were assigned by the authors using an explicit written rubric.

**Strong**
The resume role is substantially related to the JD, several important JD skills are present, and relevant experience is evident where specified.

**Partial**
Some relevant skills or role characteristics are present, but important requirements are missing.

**Poor**
The resume role is substantially unrelated to the JD and important required skills are largely absent.

_No independent recruiter annotation or inter-annotator agreement statistics were used for this benchmark._

## 13. Evaluation Metrics

Ranking metrics are treated as the primary evaluation because resume screening is fundamentally a ranking problem.

The evaluation includes:

- Recall@1
- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)
- Average rank by benchmark label

Threshold-based classification metrics are used as secondary analysis:

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted F1
- Balanced Accuracy
- Confusion Matrix

Recall@K and MRR were calculated for the three benchmark JDs for which the benchmark contains labelled Strong pairs.

## 14. Final V3 Benchmark Results

The final V3 engine produced the following benchmark observations.

### Ranking

**Strong Average Rank**

- V2 = 6.33
- V3 = 3.67

**Poor Average Rank**

- V2 = 34.33
- V3 = 38.33

### Retrieval

- Recall@5 = 0.833
- MRR = 0.583

### Threshold-Based Classification

| Metric            | Result |
| :---------------- | :----- |
| Accuracy          | 0.500  |
| Macro Precision   | 0.525  |
| Macro Recall      | 0.500  |
| Macro F1          | 0.495  |
| Weighted F1       | 0.482  |
| Balanced Accuracy | 0.500  |

The classification results are reported as secondary evidence. The benchmark is small and author-labelled, so the results should be interpreted as a proof of concept rather than evidence of general real-world performance.

## 15. Project Structure

```text
resume_analyzer/
│
├── app/
│   ├── __init__.py
│   ├── gradio_app.py
│   ├── run_evaluation.py
│   └── test_pair.py
│
├── src/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── extract.py
│   ├── parse_jd.py
│   ├── parse_resume.py
│   ├── preprocess.py
│   └── scoring.py
│
├── notebooks/
│   └── resume_analyzer_v3_research.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── test_inputs/
│
├── outputs/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 16. Important Project Files

**`src/analyzer.py`**
Main fusion engine. It:

1. extracts resume and JD text
2. parses both documents
3. builds matching text
4. generates SBERT embeddings
5. calculates semantic similarity
6. calculates rule-based signals
7. calculates V2
8. applies V3 bonuses
9. produces the final decision and explanation

**`src/scoring.py`**
Contains the core scoring functions:

- skill coverage
- title similarity
- experience matching
- V2 formula
- role-domain bonus
- explanation generation

**`app/gradio_app.py`**
Provides the application interface for testing a new resume against a new JD.

**`app/test_pair.py`**
Used to test an individual resume-JD pair.

**`app/run_evaluation.py`**
Used to run evaluation-related processing.

**`notebooks/resume_analyzer_v3_research.ipynb`**
Research and experimentation notebook used during development. The notebook contains earlier experimental scoring approaches as well as the final V3 methodology.

_The final production implementation is maintained in the `src/` and `app/` directories. The final V3 code in the Python project is the authoritative implementation of the production method._

## 17. Research Notebook

The Colab notebook documents the development process, including:

- data processing
- structured extraction
- pair generation
- scoring experiments
- V2/V3 experimentation
- benchmark construction
- ranking evaluation
- classification analysis

Earlier experimental formulas are retained in the notebook as part of the research history. They should not be interpreted as the final production configuration. The final production configuration is implemented in `src/scoring.py` and `src/analyzer.py`.

## 18. Limitations

The current system has several limitations.

**Dataset**
The research dataset contains 62 resumes, 40 JDs, and a 24-pair author-labelled benchmark. The benchmark does not include independent recruiter annotation or inter-annotator agreement statistics.

**Development Bias**
The scoring rules and thresholds were developed within the same research workflow in which the benchmark was evaluated. A separately collected and independently labelled hold-out dataset is needed for stronger validation.

**Skill Vocabulary**
The current skill extraction uses a fixed vocabulary. This can miss:

- unseen skills
- synonyms
- abbreviations
- domain-specific terminology

**Role Coverage**
Explicit role-domain rules currently cover only selected role families. Other occupations may receive no domain bonus.

**Language**
The current dataset and processing pipeline are English-language. Multilingual matching has not been evaluated.

**Classification Thresholds**
The 0.60 and 0.75 thresholds are heuristic and are not calibrated probabilities.

## 19. Future Work

Future development can include:

- larger multi-source datasets
- independent recruiter annotation
- inter-annotator agreement analysis
- separately collected hold-out evaluation
- open-vocabulary skill extraction
- skill normalization using resources such as ESCO or O\*NET
- broader role-family coverage
- multilingual encoders
- multilingual skill dictionaries
- external validation
- improved threshold calibration

## 20. Technologies

Main technologies used include:

- Python
- Sentence Transformers
- SBERT
- all-MiniLM-L6-v2
- RapidFuzz
- PyPDF / PDF text extraction tools
- python-docx
- Gradio
- Pandas
- NumPy
- Google Colab
- VS Code
