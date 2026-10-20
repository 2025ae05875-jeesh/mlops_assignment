"""Step 1: download -> validate -> clean -> save + manifest.

Usage:  python scripts/prepare_data.py [--force]
"""

import argparse
import json

from heartml import config
from heartml.data import clean, download_raw, load_raw, validate, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download")
    args = parser.parse_args()

    raw_path = download_raw(force=args.force)
    raw = load_raw(raw_path)
    report = validate(raw)
    print("[validate]", json.dumps(report, indent=2))

    cleaned = clean(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.CLEAN_FILE, index=False)
    manifest = write_manifest(raw_path, report, cleaned)
    print(f"[clean] saved {config.CLEAN_FILE} shape={cleaned.shape}")
    print(f"[manifest] sha256={manifest['raw_sha256']}")
    print(f"[manifest] class counts={manifest['clean_class_counts']}")


if __name__ == "__main__":
    main()
