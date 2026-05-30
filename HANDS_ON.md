# Hands-On Implementation Guide (beginner-friendly)

This walks you through building and running the whole thing, with the *why* at
each step, what you should *see*, and what to do when it *breaks*. Read
`OVERVIEW.md` first for the big picture.

Legend:  💻 = runs on your Mac (free)   🖥️ = runs on the rented GPU   ⏱️ = rough time

Throughout, `PY=.venv/bin/python` (the project's virtual environment).

---

## Part 0 — Sanity check the foundation 💻 ⏱️ 1 min

Everything below depends on the core modules being correct. They ship with
built-in tests. Run them:

```bash
cd "/Users/peidi/GUI/GUI 0530"
cd src && for m in coords action_space schema prompt eval_core model; do
  ../.venv/bin/python $m.py >/dev/null && echo "ok: $m"; done
cd .. && .venv/bin/python tests/test_convert.py
```

**You should see** `ok:` for each module and `test_convert.py: all tests passed`.

**What you just verified:** coordinate conversion, the action grammar, the data
schema, the prompt format, and the scoring rules all work. These are the parts
that, if wrong, waste a week — so we test them before touching a GPU.

---

## Part 1 — See the data pipeline with fake data 💻 ⏱️ 1 min

Before downloading anything, generate a tiny fake dataset and convert it to
training format, so you can *see* exactly what the model will be trained on.

```bash
.venv/bin/python scripts/make_synthetic_demo.py
.venv/bin/python src/make_sharegpt.py data/synthetic/steps.jsonl data/synthetic/sharegpt.json
```

**Look at the output:** open `data/synthetic/sharegpt.json`. One row =
one training example: a `system` message (the rules), a `user` message
(`<image>` + goal + history + "what's next?"), and an `assistant` message
(`Thought: … Action: click(500, 283)`). Open `data/synthetic/imgs/ep1_s1.png` —
the click coordinate points at the highlighted button, in 0–1000 space.

**Mental model locked in:** training data is just chat turns where the answer is
one action. That's the entire "distillation" — SFT on these rows.

---

## Part 2 — Base-model baseline on your Mac 💻 ⏱️ ~1 hr (first run downloads ~2 GB)

This is your **scientific control**: how good is the *untrained* 3B model? You
must know this before training, or you can't claim training helped.

```bash
.venv/bin/pip install mlx-vlm datasets
.venv/bin/python src/eval_screenspot.py --backend mlx --limit 200 \
  | tee results/base_screenspot.txt
```

**You should see** a line like `ScreenSpot-V2 grounding accuracy: 0.6xx` plus a
per-type breakdown. (The base Qwen2.5-VL-3B is already a decent grounder, so
expect a fairly high number here — that's fine and expected.)

**Troubleshooting**
- *Very low accuracy (~0)?* Almost always the **bbox format**. Re-run with
  `--bbox_format xyxy`. Then sanity-check: print one sample's bbox and image size
  and confirm the box sits over the right element.
- *`KeyError` on a column* (e.g. `instruction`/`bbox`/`image`)? The HF mirror uses
  different column names. Try a different `--hf_name` (e.g.
  `os-copilot/ScreenSpot-v2`) or tell me the columns and I'll adapt `_get(...)`.
- *mlx `TypeError` in generate/apply_chat_template?* mlx-vlm changed its API
  between versions. Check `.venv/bin/python -c "import mlx_vlm,importlib.metadata as m;print(m.version('mlx-vlm'))"`
  and adjust the two flagged lines in `src/model.py` (`MLXModel.generate`).
- *Too slow?* Lower `--limit` to 50 for a quick read; raise later.

Record the number. This is row 1 of your results table.

---

## Part 3 — Get real training data 💻 ⏱️ ~15–40 min

AndroidControl streams from Google's **public** bucket — you don't download all
30 GB, only the episodes you ask for.

```bash
.venv/bin/pip install tensorflow
.venv/bin/python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs \
  --out     data/androidcontrol/steps_train.jsonl \
  --max_episodes 800
```

**You should see** `episodes kept=… dropped=… | steps=… -> …steps_train.jsonl`.
A handful dropped (unsupported action types) is normal.

**Then build training rows** (absolute image paths — important for the trainer):
```bash
.venv/bin/python src/make_sharegpt.py \
  data/androidcontrol/steps_train.jsonl \
  data/androidcontrol/sharegpt_train.json --abs
```

**Also build a separate TEST slice** for the multi-step eval:
```bash
.venv/bin/python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs_test \
  --out     data/androidcontrol/steps_test.jsonl \
  --max_episodes 200
```

**Troubleshooting**
- *`gs://` permission/credentials error?* The bucket is public, but if your
  network blocks anonymous GCS, install gcloud and `gsutil -m cp` the files once
  into `data/androidcontrol/raw/`, then point `--tfrecords` at the local glob
  (see `DATA.md`).
- *TensorFlow won't install on the Mac?* Do the conversion on the GPU box instead
  (the runbook does exactly this) — nothing else in Part 3 needs your Mac.

**Sanity check before training:** spot-check that conversion is correct by
running the offline eval on the TRAIN data with the dummy model — it just
confirms files load and prompts build (accuracy will be ~0, that's fine):
```bash
.venv/bin/python src/eval_androidcontrol.py --steps data/androidcontrol/steps_train.jsonl \
  --backend dummy --limit 20
```

---

## Part 4 — Train the student (the $20 GPU burst) 🖥️ ⏱️ ~2–4 hr ≈ $1–3

You can't train a 3B VLM on the Mac — rent a Linux box with one NVIDIA GPU
(RTX 4090 is plenty, ~$0.40/hr on RunPod/Vast).

**4a. Get your code + data onto the box.** Easiest: push this repo to a private
GitHub repo and `git clone` on the box; `scp` the `data/androidcontrol/` folder
(steps + images) separately since it's gitignored. Or just regenerate the data on
the box (Part 3 commands work there too — Linux is happier with TensorFlow).

**4b. Follow `scripts/gpu_runbook.sh` block by block.** Don't blind-run the whole
file — paste one block, watch it finish, paste the next. The blocks:
0. Make a venv, install `requirements-gpu.txt` + LLaMA-Factory.
1. Build/registr the dataset (`cp configs/dataset_info.json data/androidcontrol/`).
2. **Smoke test:** train on 200 samples / 1 epoch first. **Watch the loss go
   down** and check `out/qwen3b-trackA-lora/` contains adapter files. This catches
   format bugs in 5 minutes instead of after a 2-hour run.
3. Full train, then evaluate base & student on both benchmarks.

**The most common training error** is the image path. Symptom: errors about
missing image files or zero images. Fix: you built `sharegpt_train.json` with
`--abs` (Part 3), and `data/androidcontrol/dataset_info.json` exists. If paths
still don't resolve, regenerate `sharegpt_train.json` *on the box* with `--abs`
so the absolute paths point at the box's filesystem.

**4c. Evaluate** (already in the runbook). For each of base and student:
```bash
B="--backend hf --model_path Qwen/Qwen2.5-VL-3B-Instruct"
.venv/bin/python src/eval_screenspot.py     $B [--adapter out/qwen3b-trackA-lora] --limit 400
.venv/bin/python src/eval_androidcontrol.py $B [--adapter out/qwen3b-trackA-lora] \
  --steps data/androidcontrol/steps_test.jsonl --out results/student_ac_preds.jsonl
```
(Run once without `--adapter` for base, once with it for the student.)

**4d. Pull results back and TERMINATE the instance** (stop paying):
```bash
# on your Mac:
scp -r user@box:~/repo/results ./
scp -r user@box:~/repo/out/qwen3b-trackA-lora ./out/
```

---

## Part 5 — Read the result; this is the science 💻

Fill the table from your `results/*.txt`:
```
                       ScreenSpot-V2   AndroidControl-test
base Qwen2.5-VL-3B          __              __
student (Track A)           __              __
```

**What to look for:** does the student improve a lot on **AndroidControl**
(it was trained on that distribution) while **ScreenSpot** moves less? And how
big is the remaining gap to a strong teacher on the multi-step number vs. the
grounding number? That divergence is the phenomenon your whole research question
is about — now measured by you, not read from someone's paper.

**Dig into failures** (this is where a finding lives): open
`results/student_ac_preds.jsonl` and look at the `correct: false` rows. Categorize
*why* each is wrong — wrong action type? right type, wrong coordinate? gave up
early? The per-action-type breakdown the eval already prints points you at which
action class is weakest.

---

## Part 6 — Toward the paper (next experiments)

Once Track A works end to end, the research contributions come from:

1. **Track B — actual distillation.** Run UI-TARS-7B (open, on the same GPU) over
   the AndroidControl screenshots/goals to generate its *own* `Thought:/Action:`
   traces. Train a student on those instead of human demos. Compare A vs B: how
   much does the student inherit the teacher's mistakes?
   *(Implementation: a `run_teacher.py` that loops the teacher over steps and
   writes a `steps_trackB.jsonl` in the same schema — then everything downstream
   is unchanged. Ask me to build this when you're ready.)*

2. **The black-box premise (the paper's core).** Swap the open teacher for a
   genuinely closed API. Handle that its coordinate outputs are weaker and its
   format differs. Measure the cost of the black-box constraint specifically.

3. **Stage 5 — rejection sampling.** Make two students: one on all teacher traces,
   one on only the traces that succeeded. Watch how filtering moves the static vs.
   interactive numbers differently.

4. **(Stretch) AndroidWorld.** On a proper x86+KVM Linux host, wire the student
   into the live emulator for true interactive success rate — the strongest
   multi-step evidence.

---

## Quick command reference

| Goal | Command |
|---|---|
| Test the core | `cd src && ../.venv/bin/python eval_core.py` (etc.) |
| Make fake data | `.venv/bin/python scripts/make_synthetic_demo.py` |
| Convert demos→training | `.venv/bin/python src/make_sharegpt.py IN.jsonl OUT.json --abs` |
| Convert AndroidControl | `.venv/bin/python src/convert_androidcontrol.py --tfrecords '…' --img_dir … --out … --max_episodes N` |
| Grounding eval | `.venv/bin/python src/eval_screenspot.py --backend mlx --limit 200` |
| Step eval | `.venv/bin/python src/eval_androidcontrol.py --backend hf --adapter … --steps …` |
| Train | `llamafactory-cli train configs/train_qwen3b_lora.yaml` |

When something breaks, paste the command + the error to me and I'll fix the code.
