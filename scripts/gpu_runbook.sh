#!/usr/bin/env bash
# The whole paid GPU burst, start to finish. Rent 1x RTX 4090 (Linux), then run
# these blocks. Total ~2-4 hr ~= $1-3. Read each block; don't blind-run.
set -euo pipefail

# ── 0. environment ────────────────────────────────────────────────────────────
# (assumes you cloned this repo to the box and cd'd into it)
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
pip install tensorflow                              # for reading AndroidControl
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e "LLaMA-Factory[torch,metrics]"

# ── 1. build training data (Track A: AndroidControl human demos) ───────────────
python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs \
  --out     data/androidcontrol/steps_train.jsonl \
  --max_episodes 800
# ABSOLUTE image paths so LLaMA-Factory resolves them no matter what:
python src/make_sharegpt.py \
  data/androidcontrol/steps_train.jsonl \
  data/androidcontrol/sharegpt_train.json --abs
# register the dataset where dataset_dir expects it:
cp configs/dataset_info.json data/androidcontrol/dataset_info.json

# also build a TEST slice for offline step-accuracy eval. --skip_episodes 800
# makes it DISJOINT from the 800 training episodes (no leakage -> trustworthy
# multi-step number):
python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs_test \
  --out     data/androidcontrol/steps_test.jsonl \
  --skip_episodes 800 --max_episodes 200
# NOTE: for publishable numbers, restrict train vs test to the OFFICIAL
# episode_id splits (see DATA.md) instead of just slicing different counts.

# ── 2. SMOKE TEST the training (200 samples, 1 epoch) ──────────────────────────
# Edit configs/train_qwen3b_lora.yaml: uncomment max_samples:200 / num_train_epochs:1
llamafactory-cli train configs/train_qwen3b_lora.yaml
# confirm: loss goes down, out/qwen3b-trackA-lora/ has adapter files. THEN:
# re-comment the smoke lines and run the full train:
# llamafactory-cli train configs/train_qwen3b_lora.yaml

# ── 3. evaluate: base, then student (the payoff table) ─────────────────────────
B="--backend hf --model_path Qwen/Qwen2.5-VL-3B-Instruct"
# base model:
python src/eval_screenspot.py      $B --limit 400 | tee results/base_screenspot.txt
python src/eval_androidcontrol.py  $B --steps data/androidcontrol/steps_test.jsonl \
  --out results/base_ac_preds.jsonl | tee results/base_ac.txt
# trained student (add the adapter). --coord_space norm: the student was trained
# on 0-1000 labels, so it emits normalized coords (base emits pixels -> default).
python src/eval_screenspot.py      $B --adapter out/qwen3b-trackA-lora --limit 400 \
  --coord_space norm | tee results/student_screenspot.txt
python src/eval_androidcontrol.py  $B --adapter out/qwen3b-trackA-lora --coord_space norm \
  --steps data/androidcontrol/steps_test.jsonl \
  --out results/student_ac_preds.jsonl | tee results/student_ac.txt

# ── 4. pull results back to your Mac, then TERMINATE the instance ──────────────
# (on Mac)  scp -r box:~/repo/results ./   ;  scp -r box:~/repo/out/qwen3b-trackA-lora ./out/
echo "done. remember to terminate the GPU instance."
