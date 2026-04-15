#!/bin/bash
# Test script for refactored compression pipeline
# Tests 3 models: DistilBERT, DistilGPT2, Qwen3
# All tests include: Quantization + Pruning + Fine-tuning with Knowledge Distillation

set -e  # Exit on first error

VENV_PYTHON="/home/ubuntu/MetaPAC_DEV/.venv/bin/python"
METAPAC_DIR="/home/ubuntu/MetaPAC_DEV"

echo "======================================================================"
echo "Testing Refactored Compression Pipeline"
echo "======================================================================"
echo ""
echo "Testing 3 models with FULL pipeline (Quant + Prune + FT+KD):"
echo "  1. DistilBERT (SST-2)"
echo "  2. DistilGPT2 (IMDB)"
echo "  3. Qwen3 0.6B (WOS)"
echo ""
echo "======================================================================"

cd "$METAPAC_DIR"

# Test 1: DistilBERT
echo ""
echo "======================================================================"
echo "TEST 1/3: DistilBERT SST-2"
echo "======================================================================"
$VENV_PYTHON -m metapac --config metapac/configs/compress_distilbert_sst2.yaml
EXIT_CODE_1=$?
echo "DistilBERT test exit code: $EXIT_CODE_1"

# Test 2: DistilGPT2
echo ""
echo "======================================================================"
echo "TEST 2/3: DistilGPT2 IMDB"
echo "======================================================================"
$VENV_PYTHON -m metapac --config metapac/configs/compress_distilgpt2_imdb_fast.yaml
EXIT_CODE_2=$?
echo "DistilGPT2 test exit code: $EXIT_CODE_2"

# Test 3: Qwen3
echo ""
echo "======================================================================"
echo "TEST 3/3: Qwen3 0.6B WOS"
echo "======================================================================"
$VENV_PYTHON -m metapac --config metapac/configs/compress_qwen3_wos_fast.yaml
EXIT_CODE_3=$?
echo "Qwen3 test exit code: $EXIT_CODE_3"

# Summary
echo ""
echo "======================================================================"
echo "TEST SUMMARY"
echo "======================================================================"
echo "DistilBERT: $([ $EXIT_CODE_1 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "DistilGPT2: $([ $EXIT_CODE_2 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "Qwen3:      $([ $EXIT_CODE_3 -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""

# Overall result
if [ $EXIT_CODE_1 -eq 0 ] && [ $EXIT_CODE_2 -eq 0 ] && [ $EXIT_CODE_3 -eq 0 ]; then
    echo "✓ ALL TESTS PASSED - Refactored pipeline works correctly!"
    exit 0
else
    echo "✗ SOME TESTS FAILED - Check logs above for details"
    exit 1
fi
