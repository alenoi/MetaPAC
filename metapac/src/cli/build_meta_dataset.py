# --- src path bootstrap (robust CLI) ---
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# ---------------------------------------

import argparse

from metapac.src.feature_extraction.builder import build_meta_dataset, load_config


def main():
    ap = argparse.ArgumentParser(description="MetaPAC feature extraction pipeline")
    ap.add_argument("--input", required=True, help="Bemeneti mappa (pl. artifacts)")
    ap.add_argument("--config", required=True, help="Konfigurációs YAML")
    ap.add_argument("--out", required=True, help="Kimeneti mappa (pl. artifacts/meta_dataset)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = build_meta_dataset(args.input, args.out, cfg)
    print(f"OK: meta-dataset -> {out_path}")


if __name__ == "__main__":
    main()
