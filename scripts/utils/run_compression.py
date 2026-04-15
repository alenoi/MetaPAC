"""
Run compression on DistilBERT model with rank-aware quantization.
"""
import sys
from pathlib import Path

import yaml

# Add metapac to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metapac.src.compression import run_compression


def main():
    # Load config
    config_path = Path("metapac/configs/compress_distilbert_sst2.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("COMPRESSION PIPELINE - DistilBERT SST-2")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Target model: {config['compression']['target_model']}")
    print(f"Meta checkpoint: {config['compression']['meta_checkpoint']}")
    print(f"Output: {config['compression']['output_dir']}")
    print("=" * 80)
    print()

    # Run compression
    exit_code = run_compression(config)

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ COMPRESSION SUCCESSFUL!")
        print("=" * 80)

        # Show results
        output_dir = Path(config['compression']['output_dir'])

        print(f"\n📁 Results saved to: {output_dir}")
        print(f"\n📊 Check the following files:")
        print(f"   - {output_dir}/compressed/model_state.pt")
        print(f"   - {output_dir}/compressed/quant_meta.json")
        print(f"   - {output_dir}/compressed/compression_summary.json")
        print(f"   - {output_dir}/parameter_importance_scores.csv")
        print(f"   - {output_dir}/parameter_zones.csv")
    else:
        print("\n" + "=" * 80)
        print("❌ COMPRESSION FAILED")
        print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
