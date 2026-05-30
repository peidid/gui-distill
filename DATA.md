# Getting the real data

You do NOT need to download everything. For the Track-A pilot, stream the first
few hundred episodes straight from Google's public bucket.

## AndroidControl (training + offline test eval)

Source: `gs://gresearch/android_control/android_control*` (public GZIP TFRecords).
Official schema + splits: https://github.com/google-research/google-research/tree/master/android_control

**One-time install (Mac is fine for a few-hundred-episode pilot):**
```bash
./.venv/bin/pip install tensorflow        # ~500 MB; only needed for reading TFRecords
```

**Convert a pilot slice (streams from GCS, writes only what it reads):**
```bash
./.venv/bin/python src/convert_androidcontrol.py \
  --tfrecords 'gs://gresearch/android_control/android_control*' \
  --img_dir data/androidcontrol/imgs \
  --out     data/androidcontrol/steps_train.jsonl \
  --max_episodes 500
```
~500 episodes ≈ a few thousand steps ≈ <1 GB of screenshots. Then:
```bash
./.venv/bin/python src/make_sharegpt.py \
  data/androidcontrol/steps_train.jsonl data/androidcontrol/sharegpt_train.json
```

> Network note: if `gs://` access is blocked in your environment, download the
> files once with `gsutil -m cp 'gs://gresearch/android_control/*' data/androidcontrol/raw/`
> (needs the gcloud SDK) and point `--tfrecords` at `data/androidcontrol/raw/android_control*`.

### Train/test split (matters for publishable numbers)
The TFRecords are not pre-split; AndroidControl defines train/val/test by
`episode_id`. For the pilot, any slice is fine. For the paper, filter episodes to
the official test `episode_id` list (in the google-research repo) so your
AndroidControl-test numbers are comparable to published work. We wire this into
the eval step.

## ScreenSpot-V2 (static grounding eval)

Source (HF): `os-copilot/ScreenSpot-v2` or `HongxinLi/ScreenSpot_v2` — image +
instruction + ground-truth element bbox (in pixels). Small (~1.2k samples), no
emulator. We download + score it in the eval step (`src/eval_screenspot.py`).

## AndroidWorld (interactive) — DEFERRED
Needs an x86 host + KVM for the live emulator. Skipped for the learning run; the
AndroidControl test split is our multi-step proxy. Revisit with a proper Linux box.
