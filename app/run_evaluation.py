from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import run_full_evaluation


def main() -> None:
    all_pairs_csv = PROJECT_ROOT / "outputs" / "rankings" / "all_pair_scores_v3.csv"
    benchmark_csv = PROJECT_ROOT / "outputs" / "evaluation" / "trusted_benchmark_subset_v1.csv"

    if not all_pairs_csv.exists():
        raise FileNotFoundError(f"Missing: {all_pairs_csv}")
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Missing: {benchmark_csv}")

    results = run_full_evaluation(all_pairs_csv, benchmark_csv)

    print("\n=== Ranking Summary ===")
    print(results["ranking_summary"].to_string(index=False))

    print("\n=== Ranking Diagnostics ===")
    print(results["ranking_diagnostics"].to_string(index=False))

    print("\n=== Overall Retrieval ===")
    print(results["overall_retrieval"].to_string(index=False))

    print("\n=== Confusion Matrix ===")
    print(results["confusion_matrix"].to_string())

    print("\n=== Classification Metrics ===")
    print(results["classification_metrics"].to_string(index=False))

    print("\n=== Classification Report ===")
    print(results["classification_report"])

    print("\n=== Fit Status ===")
    print(results["fit_status"].to_string(index=False))


if __name__ == "__main__":
    main()