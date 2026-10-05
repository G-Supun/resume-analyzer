from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer import analyze_new_resume_vs_new_jd


def main() -> None:
    test_dir = PROJECT_ROOT / "test_inputs"

    if len(sys.argv) >= 3:
        resume_path = Path(sys.argv[1])
        jd_path = Path(sys.argv[2])
    else:
        resume_path = test_dir / "R00T.pdf"
        jd_path = test_dir / "J00T.pdf"

    print("Resume exists:", resume_path.exists(), resume_path)
    print("JD exists    :", jd_path.exists(), jd_path)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    result = analyze_new_resume_vs_new_jd(resume_path, jd_path)

    print("\n=== FINAL ANALYZER OUTPUT ===\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()