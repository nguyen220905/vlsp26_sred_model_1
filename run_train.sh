#!/bin/bash
# Retrain the submitted model (= model A) from the official pre-trained checkpoints + the organisers' training data (~10 h on one
# 16 GB GPU). Shipped checkpoints / targets are renamed *.shipped.* before being retrained (never deleted).
# Needs: data[ban_to_chuc]/vlsp/{train.txt,data/}; data[ban_to_chuc]/public_test/{public-test.txt,data/} (each run only
# REPORTS public EER at the end; public labels are never used for any choice); ffmpeg in PATH.
set -e
cd "$(dirname "$0")"
export VLSP_ROOT="$PWD"
export TORCH_HOME="$PWD/pretrained"
PY=${PY:-python}
D="distill[12,7m_du_lieu_ban_to_chuc]"
mkdir -p logs
keep() { [ -f "$1" ] && mv "$1" "${1%.*}.shipped.${1##*.}" || true; }
step() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a logs/run_train.log; }

# 0. The fixed speaker split (train / dev-A / dev-B = 1702 / 99 / 99) and dev trial lists are in data[ban_to_chuc]/lists
#    (stages s0-s4 are NOT re-run).
step "s5 augmentation assets (training audio only + simulated RIR (pyroomacoustics) + codecs (ffmpeg))"
"$PY" -u stages/s5_aug_assets.py > logs/s5_aug_assets.log 2>&1
step "s6 simulated-edge dev trials (dev-A / dev-B evaluation only)"
"$PY" -u stages/s6_simedge.py > logs/s6_simedge.log 2>&1

step "teacher: fine-tune ReDimNet2-B6 multi on the 1702 training speakers (2 ep) - training only"
keep "$D/teachers/ft_redimnet2_b6_multi/model_final.pt"
(cd "$D/teachers/ft_redimnet2_b6_multi" && "$PY" -u run.py > console.log 2>&1)
step "KD targets: clean full-utterance teacher embeddings of every training utterance"
keep "data[ban_to_chuc]/cache/teachers/b6_ft.npz"
"$PY" -u "$D/make_targets.py" b6_ft redimnet2_b6_multi distill/teachers/ft_redimnet2_b6_multi/model_final.pt > logs/targets_b6_ft.log 2>&1

step "twin (1702 speakers, dev-A/dev-B held out): ReDimNet-M + projection KD, 4 ep -> dev-A results"
keep "$D/kd_M_b6ft_proj_4ep/model_final.pt"; keep "$D/kd_M_b6ft_proj_4ep/model_ema.pt"
(cd "$D/kd_M_b6ft_proj_4ep" && "$PY" -u run.py > console.log 2>&1)

step "model A (submitted): ReDimNet-M + projection KD, ALL 1900 speakers, 4 ep, EMA"
keep "$D/final/M_proj_alldata/model_final.pt"
(cd "$D/final/M_proj_alldata" && FINAL_EPOCHS=4 "$PY" -u run.py > console.log 2>&1)
step "DONE -> ./run_infer.sh"
