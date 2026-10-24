# VLSP 2026 SRED — Speaker Recognition on Edge Device — team ML12 (AIHub: PhamQuangNguyen)

Source code, checkpoints and reproduction instructions of the system that team ML12 re-submitted (one model, also our
public-test system) under the organisers' supplementary rules of 23/10/2026 (no transductive scoring / score
normalisation, no public-test labels for model optimisation, hyper-parameters chosen on a development split of the
training data).

> **Tóm tắt (VI):** một model ReDimNet-M 4,811,745 tham số (model A): fine-tune AAM + projection KD từ teacher
> ReDimNet2-B6 (teacher chỉ dùng lúc train), toàn bộ 1,900 speaker BTC, 4 epoch, trọng số EMA. Điểm = cosine của
> 2 embedding, mỗi file được trích riêng (batch 1) — không chuẩn hoá điểm, không dùng file test khác, không dùng nhãn
> Public. Model và siêu tham số chọn trên dev-A (speaker tách từ tập train). `./run_infer.sh` tái tạo **trùng bit** file
> điểm đã gửi BTC (private + public).

| | |
|---|---|
| Inference model | **ReDimNet-M multi** (official checkpoint from the organisers' list of 03/10), **4,811,745 parameters** (< 7M), one model |
| Model A | ReDimNet-M fine-tuned with AAM-softmax (m 0.2, s 32) + projection knowledge distillation (linear 192→192 layer, training only; loss `2.0·(1 − cos)`) from a fine-tuned ReDimNet2-B6 multi teacher (official list; **training only**), all 1,900 training speakers, 3 s crops, 4 epochs, EMA 0.999 checkpoint |
| Data | **Only the training data provided by the organisers.** Augmentation: non-speech segments of the training audio itself, simulated RIR (pyroomacoustics), codecs (ffmpeg) |
| Scoring | **cosine** of the two L2-normalised embeddings. Every test file is embedded **alone** (batch of one, whole file, no crop/pad) → a trial's score depends on its two files only. No cohort, no score normalisation, no calibration |
| Selection | recipe and epoch count fixed on 05/10/2026 on dev-A (`distill[...]/final/decide_epochs.py`: 4 epochs beat 2 epochs on pooled dev-A; section 6). Public labels never used |

| set | EER (%) | note |
|---|---|---|
| dev-A pooled (1,702-speaker twin, 99 held-out speakers, 38,000 trials) | 3.811 | selection metric |
| dev-B pooled (twin, 99 other held-out speakers, 38,000 trials) | 2.642 | |
| public test (49,993 trials) | 5.029 (minDCF 0.312) | reporting only, computed after the choice |
| private test | — | scored by the organisers |

## 1. Setup
Tested environment: Windows 10 + Git Bash, Python 3.11.9, PyTorch 2.11.0 + CUDA 12.8, NVIDIA RTX 5060 Ti 16 GB (driver
591.74). The code is plain Python/bash and also runs on Linux. **A CUDA GPU is required.** Inference needs < 2 GB of GPU
memory; training uses ~10–12 GB. The repository is ~250 MB (checkpoints included, no Git LFS needed).
```bash
git clone https://github.com/nguyen220905/vlsp26_sred_model_1.git
cd vlsp26_sred_model_1
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
- Run the `.sh` scripts with bash (on Windows: Git Bash). `PY=python` selects the Python interpreter of the environment.
- `common/config.py` uses the repository folder as root (or the `VLSP_ROOT` environment variable); the scripts set
  `TORCH_HOME=./pretrained`, so the two official checkpoints shipped in `pretrained/hub/checkpoints/` are used and nothing
  is downloaded. The ReDimNet / ReDimNet2 architecture code is vendored in `third_party/`.
- Folder names contain `[ ]` and `,` (kept as they were during the experiments) — quote them in a shell.

## 2. Data (not redistributed — copy the organisers' files here)
```
data[ban_to_chuc]/private_test/private-test.txt    trial list, one pair per line: "id07900/0010.wav id05180/0016.wav"
data[ban_to_chuc]/private_test/data/idXXXXX/*.wav  private-test audio (paths in the list are relative to data/)
data[ban_to_chuc]/public_test/public-test.txt      "wav1 wav2 target|nontarget"  (only for ./run_infer.sh public)
data[ban_to_chuc]/public_test/data/...             public-test audio
data[ban_to_chuc]/vlsp/train.txt                   "id00000<TAB>id00000/0009.wav"  (training list; only for run_train.sh)
data[ban_to_chuc]/vlsp/data/idXXXXX/*.wav          training audio (unzipped Vi-SRED26-Train.zip)
```
If the files are elsewhere, pass them explicitly: `PY=python ./run_infer.sh <trial list> <wav folder> <output name>`.
`data[ban_to_chuc]/lists/` holds our fixed speaker split (train / dev-A / dev-B = 1702 / 99 / 99) and the dev trial lists.

## 3. Reproduce the submitted score files (no training)
```bash
PY=python ./run_infer.sh            # private test: ~75 min on one GPU (every file embedded alone, ~8 files/s)
PY=python ./run_infer.sh public     # public test:  ~35 min
```
What the script does (`common/predict.py --score cos --per_file`): loads `distill[...]/final/M_proj_alldata/model_final.pt`,
asserts that the model has fewer than 7M parameters (4,811,745), embeds every file of the trial list alone,
writes the cosine score of each pair in the order of the trial list, checks the file and zips it.
Output: `outputs/<name>/results.txt` (`file1 file2 score`) and `outputs/<name>/results.zip` (AIHub format). Expected end
of the log for the private test:
```
OK .../outputs/private/results.txt: 278410 lines, score range [-0.286, 0.997]
a5995ef4d4a37f9dd5df975100c6be904bbf7c93 *outputs/private/results.txt
sha1 of the submitted results.txt:   a5995ef4d4a37f9dd5df975100c6be904bbf7c93
```
| file | sha1 of `results.txt` |
|---|---|
| private (`submitted/private_results.zip`) | a5995ef4d4a37f9dd5df975100c6be904bbf7c93 |
| public (`submitted/public_results.zip`, best system) | 5b4e088c6c2120dcfcb1a9d725222e859fa275e1 (EER 5.029 % is printed, the public list has labels) |

**Verified 25/10/2026** from a fresh `git clone` of this repository (test data linked into `data[ban_to_chuc]/`): `./run_infer.sh` and `./run_infer.sh public` reproduce both submitted files **bit-exactly** (sha1 above) on the tested environment of section 1.

**Bit-exactness depends on the hardware/software stack.** On another GPU model, driver or CUDA/cuDNN version, the last
digits of the floating-point scores can differ, so the sha1 can differ although the system is the same. In that case
compare the files directly:
```bash
python tools/compare_scores.py outputs/private/results.txt submitted/private_results.zip
python tools/compare_scores.py outputs/public/results.txt submitted/public_results.zip "data[ban_to_chuc]/public_test/public-test.txt"
```
It checks that both files list the same trials in the same order and prints the maximum score difference, the Spearman
rank correlation (expected ≈ 1.0) and, for the public test, the EER of both files (expected 5.029 %).

## 4. Retrain everything from the official checkpoints (~7.5 h on one 16 GB GPU)
```bash
PY=python ./run_train.sh                  # everything, logs in logs/
SKIP_TEACHER=1 PY=python ./run_train.sh   # reuse the shipped fine-tuned teacher + KD targets (~4.7 h)
PY=python ./run_infer.sh
```
The fine-tuned teacher (`distill[...]/teachers/ft_redimnet2_b6_multi/model_final.pt`) and its KD targets
(`data[ban_to_chuc]/cache/teachers/b6_ft.npz`) are shipped, so model A can be retrained without retraining the teacher.

| step | script | time |
|---|---|---|
| augmentation assets | `stages/s5_aug_assets.py` (noise = non-speech parts of training audio; simulated RIR; 25 % of training utterances through codecs) | ~5 min |
| simulated-edge dev trials | `stages/s6_simedge.py` (dev-A / dev-B evaluation only) | ~5 min |
| teacher | `distill[...]/teachers/ft_redimnet2_b6_multi/run.py`: ReDimNet2-B6 multi, AAM, 3 s crops, 2 epochs, 1,702 training speakers | ~2.3 h |
| KD targets | `distill[...]/make_targets.py`: clean full-utterance teacher embeddings of every training utterance | ~0.6 h |
| twin (1,702 spk) | `distill[...]/kd_M_b6ft_proj_4ep/run.py`: same recipe, 4 epochs, dev-A + public evaluation | ~2.6 h |
| model A (submitted) | `distill[...]/final/M_proj_alldata/run.py` (`FINAL_EPOCHS=4`), all 1,900 speakers, EMA | ~2 h |

Each training run writes `results.json` (dev-A EER of the twin, public EER for information) and its checkpoints in its
own folder. GPU training is not bit-deterministic; inference from a given checkpoint is. A full retraining of model A with the same
recipe (16/10/2026) gave public-test EER 5.033 with cosine scoring vs 5.013 for the shipped checkpoint (same
length-batched evaluation), i.e. within the seed-to-seed noise of about 0.1. Shipped checkpoints/targets are renamed
`*.shipped.*` (not deleted) before retraining. Model A was trained with an earlier revision of `common/train.py`
(code hash `d36e74c71c6e` in its `manifest.json`); the shipped revision only adds options that are off by default.

## 5. Compliance with the rules
- < 7M parameters at inference (asserted by `common/predict.py`); only checkpoints from the organisers' list; the
  ReDimNet2-B6 teacher (12.3M) is used during training only.
- Speaker labels come only from the organisers' training data; no other speech data is used for training.
- Inference is per trial: each file is embedded alone and the score is the cosine of the two embeddings — no cohort,
  no AS-norm / score normalisation, no calibration, no query expansion, no statistic of other test files.
- Public-test labels are not used for any choice (public EER is printed for information only).
- The recipe, the epoch count and the choice of this model were made on dev-A (held-out training speakers).
- Our earlier AIHub submissions used AS-norm + a duration-aware calibration and are **not** represented by this code.

## 6. Model selection (dev only)
The recipe (projection KD from the fine-tuned B6 teacher) and the number of epochs were chosen on pooled dev-A of the
1,702-speaker twins, before any test result was used: `distill[...]/final/decide_epochs.py` picks 4 epochs if the
4-epoch twin beats the mean of the two 2-epoch seeds by more than the seed spread (0.02); the decision and its inputs are
in `distill[...]/final/M_proj_alldata/epoch_decision.json` (2-epoch seeds 3.732 / 3.747, 4-epoch 3.642 with the AS-norm
scoring used for development at that time). With the cosine scoring of the submission, the twin of A reaches 3.811
(dev-A pooled) and 2.642 (dev-B pooled): `distill[...]/kd_M_b6ft_proj_4ep/eval_ema/results*.json`, field
`devA_pooled.cos` / `devB_pooled.cos`.

## 7. Checksums (sha1)
| file | sha1 |
|---|---|
| `distill[...]/final/M_proj_alldata/model_final.pt` (**submitted model** = model A, EMA) | 1e1633838d3299dbe8282d9c1b49b9552c50e4dd |
| `distill[...]/kd_M_b6ft_proj_4ep/model_ema.pt` (twin of A) | 38b54ff25dff68a5fc250aed1e4ad09047b491b7 |
| `distill[...]/teachers/ft_redimnet2_b6_multi/model_final.pt` (teacher, training only) | 559ce282f24de91d41851c7aa90edd05082037a0 |
| `data[ban_to_chuc]/cache/teachers/b6_ft.npz` (KD targets) | d26c312dc57d1445761a76cbbdba3837b9af53ac |
| `pretrained/.../M-vb2+vox2+cnc-ft_mix.pt` (official) | 79cb491669894a2cb7e93a2001c03d56048c7db7 |
| `pretrained/.../b6-vb2+vox2+cnc2_v0-lm.pt` (official) | bac202c7801574d43d5047d239275114c2b09c95 |

## 8. Layout
```
run_infer.sh / run_train.sh     reproduce the submitted scores / retrain everything
common/                         models, training (AAM + KD), augmentation, embedding extraction (per-file mode), EER, prediction
stages/                         data preparation (s0–s4 results are already in data[...]/lists), s5 augmentation, s6 simulated-edge dev trials
third_party/redimnet, redimnet2 architecture code (IDRnD/redimnet @ ce039a62, PalabraAI/redimnet2 @ c5bbe0b7)
pretrained/hub/checkpoints/     the two official checkpoints used (ReDimNet-M multi, ReDimNet2-B6 multi)
distill[12,7m_du_lieu_ban_to_chuc]/
  teachers/ft_redimnet2_b6_multi/  fine-tuned teacher (training only)
  kd_M_b6ft_proj_4ep/              1,702-speaker twin of A (dev evaluation)
  final/M_proj_alldata/            submitted model A (1,900 speakers, 4 epochs); final/decide_epochs.py = epoch choice on dev-A
submitted/                      the exact score files sent to the organisers (private and public test)
tools/compare_scores.py         compare a reproduced score file with the submitted one (when the sha1 differs)
outputs/                        created by run_infer.sh (not in git)
```
