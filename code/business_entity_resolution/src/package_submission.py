"""Build <team>_submission.zip in the layout required by the challenge.

    python src/package_submission.py --team <team_name> [--doc path/to/Documentation_template.md]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, PACKAGE_ROOT, PROJECT_DIR

SKIP_DIRS = {"__pycache__", ".pytest_cache", "artifacts", "cache"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--doc", type=Path, default=PACKAGE_ROOT / "Documentation_template.md")
    ap.add_argument("--dest", type=Path, default=PACKAGE_ROOT)
    args = ap.parse_args()

    zpath = args.dest / f"{args.team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(args.output_dir / name, f"output/{name}")
        for p in PROJECT_DIR.rglob("*"):
            rel = p.relative_to(PROJECT_DIR)
            if p.is_file() and not SKIP_DIRS & set(rel.parts):
                z.write(p, f"code/business_entity_resolution/{rel.as_posix()}")
        z.write(args.doc, args.doc.name)
    print(f"wrote {zpath}")


if __name__ == "__main__":
    main()
