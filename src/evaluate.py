from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def score_to_class(score: float) -> int:
    if score >= 0.75:
        return 2
    elif score >= 0.60:
        return 1
    return 0


def load_and_merge_benchmark(all_pairs_csv: str | Path, benchmark_csv: str | Path) -> pd.DataFrame:
    full_df = pd.read_csv(all_pairs_csv)
    bench = pd.read_csv(benchmark_csv)

    if "rank_within_jd_v3" not in full_df.columns:
        full_df = full_df.sort_values(["jd_id", "final_score_v3"], ascending=[True, False]).reset_index(drop=True)
        full_df["rank_within_jd_v3"] = full_df.groupby("jd_id").cumcount() + 1

    if "label_code" in bench.columns:
        bench["gold_label"] = pd.to_numeric(bench["label_code"], errors="coerce")
    elif "label" in bench.columns:
        tmp = pd.to_numeric(bench["label"], errors="coerce")
        if tmp.notna().sum() > 0:
            bench["gold_label"] = tmp
        else:
            bench["gold_label"] = bench["label"].astype(str).str.lower().map({
                "poor": 0, "partial": 1, "strong": 2
            })
    elif "human_label" in bench.columns:
        bench["gold_label"] = bench["human_label"].astype(str).str.lower().map({
            "poor": 0, "partial": 1, "strong": 2
        })
    else:
        raise ValueError("No benchmark label column found.")

    bench = bench.dropna(subset=["gold_label"]).copy()
    bench["gold_label"] = bench["gold_label"].astype(int)

    merge_cols = [
        "jd_id", "resume_id", "final_score_v3", "rank_within_jd_v3",
        "semantic_score", "skill_score", "title_score", "experience_score"
    ]
    extra_cols = [c for c in ["jd_role", "resume_role"] if c in full_df.columns]
    merge_cols += extra_cols

    bench_eval = bench.merge(full_df[merge_cols], on=["jd_id", "resume_id"], how="left")
    bench_eval["pred_label_v3"] = bench_eval["final_score_v3"].apply(score_to_class)

    return bench_eval


def ranking_summary(bench_eval: pd.DataFrame) -> pd.DataFrame:
    return (
        bench_eval.groupby("gold_label")
        .agg(
            count=("resume_id", "size"),
            avg_rank=("rank_within_jd_v3", "mean"),
            median_rank=("rank_within_jd_v3", "median"),
            avg_score=("final_score_v3", "mean"),
            median_score=("final_score_v3", "median"),
        )
        .reset_index()
        .sort_values("gold_label", ascending=False)
    )


def ranking_diagnostics(bench_eval: pd.DataFrame) -> pd.DataFrame:
    strong_top1 = int(((bench_eval["gold_label"] == 2) & (bench_eval["rank_within_jd_v3"] == 1)).sum())
    strong_top3 = int(((bench_eval["gold_label"] == 2) & (bench_eval["rank_within_jd_v3"] <= 3)).sum())
    strong_top5 = int(((bench_eval["gold_label"] == 2) & (bench_eval["rank_within_jd_v3"] <= 5)).sum())
    strong_gt10 = int(((bench_eval["gold_label"] == 2) & (bench_eval["rank_within_jd_v3"] > 10)).sum())

    poor_top10 = int(((bench_eval["gold_label"] == 0) & (bench_eval["rank_within_jd_v3"] <= 10)).sum())
    poor_bottom20 = int(((bench_eval["gold_label"] == 0) & (bench_eval["rank_within_jd_v3"] >= 20)).sum())

    return pd.DataFrame([
        ["Strong matches in Top-1", strong_top1],
        ["Strong matches in Top-3", strong_top3],
        ["Strong matches in Top-5", strong_top5],
        ["Strong matches ranked > 10", strong_gt10],
        ["Poor matches ranked <= 10", poor_top10],
        ["Poor matches ranked >= 20", poor_bottom20],
    ], columns=["metric", "value"])


def retrieval_metrics(bench_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def recall_at_k(df_jd: pd.DataFrame, k: int) -> float:
        strong = df_jd[df_jd["gold_label"] == 2]
        if len(strong) == 0:
            return np.nan
        return (strong["rank_within_jd_v3"] <= k).sum() / len(strong)

    def mrr_for_jd(df_jd: pd.DataFrame) -> float:
        strong = df_jd[df_jd["gold_label"] == 2].sort_values("rank_within_jd_v3")
        if len(strong) == 0:
            return np.nan
        first_rank = strong.iloc[0]["rank_within_jd_v3"]
        return 1.0 / first_rank

    per_jd = []
    for jd_id, g in bench_eval.groupby("jd_id", sort=True):
        per_jd.append({
            "jd_id": jd_id,
            "strong_count": int((g["gold_label"] == 2).sum()),
            "recall_at_1": recall_at_k(g, 1),
            "recall_at_3": recall_at_k(g, 3),
            "recall_at_5": recall_at_k(g, 5),
            "mrr": mrr_for_jd(g),
        })

    per_jd_df = pd.DataFrame(per_jd)
    overall_df = pd.DataFrame([{
        "avg_recall_at_1": per_jd_df["recall_at_1"].mean(),
        "avg_recall_at_3": per_jd_df["recall_at_3"].mean(),
        "avg_recall_at_5": per_jd_df["recall_at_5"].mean(),
        "avg_mrr": per_jd_df["mrr"].mean(),
    }])

    return per_jd_df, overall_df


def classification_outputs(bench_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    y_true = bench_eval["gold_label"]
    y_pred = bench_eval["pred_label_v3"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=["True_Poor", "True_Partial", "True_Strong"],
        columns=["Pred_Poor", "Pred_Partial", "Pred_Strong"]
    )

    metrics_df = pd.DataFrame([{
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }])

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=["Poor", "Partial", "Strong"],
        digits=4,
        zero_division=0
    )

    return cm_df, metrics_df, report


def fit_status_diagnostic(ranking_summary_df: pd.DataFrame, diagnostics_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    avg_rank_strong = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 2, "avg_rank"].iloc[0])
    avg_rank_partial = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 1, "avg_rank"].iloc[0])
    avg_rank_poor = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 0, "avg_rank"].iloc[0])

    avg_score_strong = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 2, "avg_score"].iloc[0])
    avg_score_partial = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 1, "avg_score"].iloc[0])
    avg_score_poor = float(ranking_summary_df.loc[ranking_summary_df["gold_label"] == 0, "avg_score"].iloc[0])

    score_gap = avg_score_strong - avg_score_poor

    strong_gt10 = int(diagnostics_df.loc[diagnostics_df["metric"] == "Strong matches ranked > 10", "value"].iloc[0])
    poor_top10 = int(diagnostics_df.loc[diagnostics_df["metric"] == "Poor matches ranked <= 10", "value"].iloc[0])

    bal_acc = float(metrics_df["balanced_accuracy"].iloc[0])
    macro_f1 = float(metrics_df["macro_f1"].iloc[0])

    if (
        avg_rank_strong <= 5 and
        avg_rank_poor >= 20 and
        strong_gt10 == 0 and
        poor_top10 == 0 and
        score_gap >= 0.20
    ):
        fit_status = "Balanced / well-separated on trusted benchmark"
    elif (
        avg_rank_strong > 10 or
        avg_rank_poor < 15 or
        score_gap < 0.10
    ):
        fit_status = "Underfitted / weak separation on benchmark"
    else:
        fit_status = "Partially balanced, but needs more checking"

    note = (
        "Classical overfitting is not directly applicable because v3 is a fixed hybrid engine. "
        "Use a new unseen manually labeled holdout set to test over-tuning."
    )

    return pd.DataFrame([{
        "avg_rank_strong": avg_rank_strong,
        "avg_rank_partial": avg_rank_partial,
        "avg_rank_poor": avg_rank_poor,
        "avg_score_strong": avg_score_strong,
        "avg_score_partial": avg_score_partial,
        "avg_score_poor": avg_score_poor,
        "score_gap_strong_poor": score_gap,
        "strong_ranked_gt10": strong_gt10,
        "poor_ranked_top10": poor_top10,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "fit_status": fit_status,
        "note": note,
    }])


def run_full_evaluation(all_pairs_csv: str | Path, benchmark_csv: str | Path) -> Dict[str, Any]:
    bench_eval = load_and_merge_benchmark(all_pairs_csv, benchmark_csv)
    rs = ranking_summary(bench_eval)
    rd = ranking_diagnostics(bench_eval)
    per_jd, overall = retrieval_metrics(bench_eval)
    cm_df, metrics_df, report = classification_outputs(bench_eval)
    fit_df = fit_status_diagnostic(rs, rd, metrics_df)

    return {
        "bench_eval": bench_eval,
        "ranking_summary": rs,
        "ranking_diagnostics": rd,
        "per_jd_retrieval": per_jd,
        "overall_retrieval": overall,
        "confusion_matrix": cm_df,
        "classification_metrics": metrics_df,
        "classification_report": report,
        "fit_status": fit_df,
    }