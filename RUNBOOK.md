# Operator runbook — do these in order

Everything up to step 2 is DONE and tested. Steps 2–5 are commands you run
(they need network / a GPU, which the build session couldn't do for you).

## ✅ 0–1. Built & verified on the Mac (no action needed)
- Action space, coords, schema, prompt, eval scoring — all self-tested.
- Synthetic demo proves data→ShareGPT→eval end to end.
- Verify anytime:
  ```bash
  cd src && for m in coords action_space schema prompt eval_core model; do ../.venv/bin/python $m.py >/dev/null && echo ok:$m; done
  cd .. && .venv/bin/python tests/test_convert.py
  ```

## 2. Base-model baseline on your Mac (free, ~1 hr, do this first)
Get the *untrained* Qwen2.5-VL-3B numbers BEFORE spending money — this is your
control. Uses mlx (Apple-Silicon native).
```bash
.venv/bin/pip install mlx-vlm datasets
# grounding baseline (downloads ~2 GB model on first run):
.venv/bin/python src/eval_screenspot.py --backend mlx --limit 200 | tee results/base_screenspot.txt
```
> First, eyeball 2–3 ScreenSpot samples to confirm `--bbox_format` (xywh vs xyxy)
> and that boxes look right. A wrong bbox format silently tanks the number.

## 3. Real training data (free, streams from Google's bucket)
See DATA.md. Short version:
```bash
.venv/bin/pip install tensorflow
.venv/bin/python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs --out data/androidcontrol/steps_train.jsonl \
  --max_episodes 800
```

## 4. The $20 GPU burst (train + full eval)
Rent 1× RTX 4090 (RunPod/Vast, Linux). Follow `scripts/gpu_runbook.sh` block by
block. It installs LLaMA-Factory, builds data, smoke-tests, trains the LoRA, and
evaluates base vs student on both benchmarks. ~2–4 hr ≈ $1–3.

## 5. The payoff: fill the table & interpret
```
                       ScreenSpot-V2   AndroidControl-test
base Qwen2.5-VL-3B          __              __
student (Track A)           __              __
teacher (UI-TARS-7B)        __              __
```
Expected story: student closes most of the *grounding* gap but lags on
*multi-step* — that divergence is your research finding. Then:
- **Track B (distillation):** run UI-TARS-7B over the same prompts → traces → SFT.
- **Stage 5 (ablation):** pure vs. rejection-sampled traces; watch how filtering
  moves static vs. interactive numbers differently.

## What this learning run is NOT (yet)
- No closed-API black-box teacher (that's the paper's premise — swap in later).
- No live AndroidWorld (needs x86+KVM; AndroidControl-test is the proxy).
- No official train/test episode split enforced (fine for learning; required for
  publishable numbers — see DATA.md).
