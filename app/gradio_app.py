from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer import analyze_new_resume_vs_new_jd


def build_match_profile_radar(result: dict):
    labels = ["Semantic", "Skill", "Title", "Experience"]
    values = [
        result.get("semantic_score", 0.0),
        result.get("skill_score", 0.0),
        result.get("title_score", 0.0),
        result.get("experience_score", 0.0),
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_cycle = values + [values[0]]
    angles_cycle = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_cycle, values_cycle, linewidth=2)
    ax.fill(angles_cycle, values_cycle, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Match Profile Radar", pad=20)
    return fig


def build_score_construction_chart(result: dict):
    contrib_sem = 0.4 * result.get("semantic_score", 0.0)
    contrib_skill = 0.3 * result.get("skill_score", 0.0)
    contrib_title = 0.2 * result.get("title_score", 0.0)
    contrib_exp = 0.1 * result.get("experience_score", 0.0)
    skill_boost = result.get("skill_boost", 0.0)
    domain_bonus = result.get("domain_bonus", 0.0)

    labels = [
        "0.4×Semantic",
        "0.3×Skill",
        "0.2×Title",
        "0.1×Experience",
        "+Skill Boost",
        "+Domain Bonus",
    ]
    steps = [contrib_sem, contrib_skill, contrib_title, contrib_exp, skill_boost, domain_bonus]
    cumulative = np.cumsum([0] + steps)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, step in enumerate(steps):
        ax.bar(i, step, bottom=cumulative[i], width=0.6)
        ax.text(i, cumulative[i] + step + 0.01, f"{step:.3f}", ha="center")

    final_v3 = result.get("final_score_v3", 0.0)
    ax.bar(len(labels), final_v3, width=0.6)
    ax.text(len(labels), final_v3 + 0.01, f"{final_v3:.3f}", ha="center")

    ax.set_xticks(range(len(labels) + 1))
    ax.set_xticklabels(labels + ["Final v3"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("How Final v3 Score Is Built")
    plt.tight_layout()
    return fig


def build_jd_skill_status_table(result: dict) -> pd.DataFrame:
    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])

    rows = []

    for skill in matched:
        rows.append({
            "JD Skill": skill,
            "Status": "Matched"
        })

    for skill in missing:
        rows.append({
            "JD Skill": skill,
            "Status": "Missing"
        })

    if not rows:
        rows.append({
            "JD Skill": "No extracted JD skill",
            "Status": "-"
        })

    return pd.DataFrame(rows)


def build_summary_text(result: dict) -> str:
    return (
        f"### Final Decision: **{result.get('decision', 'Unknown')}**\n\n"
        f"- **Final Score (v3):** {result.get('final_score_v3', 0):.4f}\n"
        f"- **Resume Role:** {result.get('resume_role', '')}\n"
        f"- **JD Role:** {result.get('jd_role', '')}\n"
        f"- **JD Type:** {result.get('jd_type', '')}\n"
        f"- **Matched Skill Count:** {result.get('matched_skill_count', 0)}\n"
        f"- **Explanation:** {result.get('explanation', '')}"
    )


def run_analysis(resume_file, jd_file):
    empty_df = pd.DataFrame([{
        "JD Skill": "",
        "Status": ""
    }])

    if resume_file is None or jd_file is None:
        return (
            {"error": "Please upload both a resume file and a job description file."},
            "### Final Decision: **Not available**",
            None,
            None,
            empty_df,
        )

    result = analyze_new_resume_vs_new_jd(resume_file, jd_file)

    summary = build_summary_text(result)
    radar = build_match_profile_radar(result)
    score_build = build_score_construction_chart(result)
    jd_skill_table = build_jd_skill_status_table(result)

    return result, summary, radar, score_build, jd_skill_table


with gr.Blocks(
    title="Resume Analyzer - Final v3 Hybrid Engine",
    delete_cache=(86400, 86400),
) as demo:
    gr.Markdown("# Resume Analyzer - Final v3 Hybrid Engine")
    gr.Markdown(
        "This dashboard shows the match profile, score construction, "
        "and JD skills status for the uploaded resume and job description."
    )

    with gr.Row():
        resume_input = gr.File(label="Upload Resume")
        jd_input = gr.File(label="Upload Job Description")

    run_btn = gr.Button("Analyze Match")

    summary_md = gr.Markdown()

    with gr.Row():
        radar_plot = gr.Plot(label="Match Profile Radar")
        score_build_plot = gr.Plot(label="Score Construction")

    jd_skill_table = gr.Dataframe(label="JD Skills: Matched / Missing", interactive=False)
    output_json = gr.JSON(label="Analyzer Output")

    run_btn.click(
        fn=run_analysis,
        inputs=[resume_input, jd_input],
        outputs=[output_json, summary_md, radar_plot, score_build_plot, jd_skill_table]
    )

if __name__ == "__main__":
    demo.launch()