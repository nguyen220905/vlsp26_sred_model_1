#!/bin/bash
# Reproduce the submitted score files (no training): ReDimNet-M model A (projection KD, 4 ep, all 1900 speakers, EMA),
# cosine scoring, every test file embedded ALONE (batch of one, whole file) -> each trial depends on its two files only.
# usage: ./run_infer.sh            private test (default)
#        ./run_infer.sh public     public test
#        ./run_infer.sh <trial list> <wav folder> <name>   any list "wav1 wav2 [label]"
# output: outputs/<name>/results.txt ("file1 file2 score", trial-list order) + results.zip (AIHub format)
set -e
cd "$(dirname "$0")"
export VLSP_ROOT="$PWD"                 # common/config.py: root = this folder
export TORCH_HOME="$PWD/pretrained"     # official pre-trained checkpoints shipped in pretrained/ (no download)
PY=${PY:-python}
CKPT=distill/final/M_proj_alldata/model_final.pt
case "${1:-private}" in
  private) TRIALS="data[ban_to_chuc]/private_test/private-test.txt"; WAV="data[ban_to_chuc]/private_test/data"; NAME=private
           REF=a5995ef4d4a37f9dd5df975100c6be904bbf7c93 ;;
  public)  TRIALS="data[ban_to_chuc]/public_test/public-test.txt"; WAV="data[ban_to_chuc]/public_test/data"; NAME=public
           REF=5b4e088c6c2120dcfcb1a9d725222e859fa275e1 ;;
  *)       TRIALS="$1"; WAV="$2"; NAME="${3:-custom}"; REF="" ;;
esac
O="outputs/$NAME"
mkdir -p "$O"
"$PY" -u -m common.predict --model redimnet_M_multi --ckpt "$CKPT" --trials "$TRIALS" --wav_dir "$WAV" \
    --score cos --per_file --out "$O/results.txt"
"$PY" -m common.predict --check "$O/results.txt" --trials "$TRIALS"
"$PY" -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED); z.write(sys.argv[2], 'results.txt'); z.close()" \
    "$O/results.zip" "$O/results.txt"
sha1sum "$O/results.txt"
if [ -n "$REF" ]; then echo "sha1 of the submitted results.txt:   $REF"; fi
