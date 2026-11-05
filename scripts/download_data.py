"""Download the LMArena arena-human-preference-140k dataset into data/raw/."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.paths import DATA_RAW

DATASET_ID = "lmarena-ai/arena-human-preference-140k"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the LMArena 140k preference dataset.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", type=Path, default=DATA_RAW)
    args = parser.parse_args()

    from datasets import load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "arena-human-preference-140k.parquet"
    metadata_path = args.out_dir / "arena-human-preference-140k.metadata.json"

    print(f"Downloading {args.dataset_id} ({args.split}) ...")
    dataset = load_dataset(args.dataset_id, split=args.split)
    df = dataset.to_pandas()
    df.to_parquet(parquet_path, index=False)

    metadata = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "parquet_path": str(parquet_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {parquet_path}")


if __name__ == "__main__":
    main()
