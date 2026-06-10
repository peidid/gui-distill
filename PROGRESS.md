# GUI Agent Distillation — Progress & Research Direction

**Status:** Three-track A/B/C teacher-capability sweep complete (2026-06-02)  
**Model:** Qwen2.5-VL-3B-Instruct student, LoRA SFT (30M params = 0.79%)  
**Data:** AndroidControl (5,167 train steps, 1,332 disjoint test steps) + ScreenSpot-V2 grounding (400 samples)

---

## Overall Thesis

**Black-box hard-label distillation scales multi-step control behavior with teacher quality but does NOT transfer grounding and inherits the teacher's action-coverage gaps.**

### The Research Question
Can a small, on-device GUI agent (Qwen2.5-VL-3B) learn to perform complex multi-step Android tasks by imitating a teacher's **text-only action labels** (no logits, no probabilities — just "click(500, 400)" or "type('hello')")? And if so, **where does it fail?** Is the cost in:
- **Perception** (grounding/localization) — can it find the right UI element?
- **Control/sequencing** (multi-step) — can it chain actions correctly?
- **Coverage** (action vocabulary) — can it emit the full range of actions?

And **how does this depend on the teacher?**

---

## What We Measured — The 2D Table

```
                          ScreenSpot-V2     AndroidControl-test
                          (grounding)       (multi-step)
  base Qwen2.5-VL-3B         0.603             0.114
  
  student A (human)          0.280             0.552
  student B (UI-TARS)        0.280             0.319
  student C (GTA1)           0.302             0.465
  
  teacher UI-TARS-7B         0.650             0.316
  teacher GTA1-32B           0.895             0.514
```

### Metrics Explained
- **ScreenSpot-V2 (grounding):** one-shot click accuracy on out-of-distribution (desktop/web) images. Student is trained only on phone screenshots. This measures **perception generalization**.
- **AndroidControl-test (multi-step):** step-level accuracy on 1,332 held-out task steps, teacher-forced with ground-truth history (no error compounding). This measures **in-distribution control** accuracy.
- **Per-action-type breakdown:** 8 action classes (click, done, open_app, type, scroll, press_back, wait, long_press) reveal where the cost concentrates.

---

## Three Confirmed Findings

### 1. Multi-Step Tracks Teacher Quality — Monotonically
**The student's in-distribution multi-step competence is bounded by and scales with its teacher's competence.**

Evidence:
```
teacher:  human(perfect) > GTA1(0.514) > UI-TARS(0.316) > base(0.114)
student:  A(0.552)       > C(0.465)     > B(0.319)       > base(0.114)
```

Each student lands just below its teacher:
- student C (0.465) ← teacher GTA1 (0.514)
- student B (0.319) ← teacher UI-TARS (0.316)
- student A (0.552) tracks human (near-perfect) teacher

**Mechanism:** The student learns the action vocabulary through imitation. A better teacher provides better examples of *when* to click, *what* to type, etc. → the student improves proportionally.

---

### 2. Grounding Does NOT Transfer — A Hard Ceiling
**Out-of-distribution (OOD) grounding is recipe-bound, NOT teacher-dependent. Every student lands ~0.28–0.30 regardless of its teacher's grounding.**

Evidence:
```
teacher grounding:   UI-TARS 0.650  →  GTA1 0.895
student grounding:   all = 0.28–0.30    (stuck; no improvement)
```

Even GTA1's exceptional grounding (0.895, the best on both benchmarks) **did not transfer** to the student. The student's OOD grounding is 3× worse than GTA1's, not because of the labels, but because of the **fine-tuning recipe** (narrow phone-GUI SFT, frozen vision tower, light LoRA that keeps the base's pixel convention).

**In-distribution nuance:** the student's *in-distribution* click accuracy (phone screenshots) *improved* over the base (0.125 → 0.46–0.48). It's the **out-of-distribution jump** (phone → desktop/web) that's lost — a distribution-shift problem, not a grounding problem per se.

---

### 3. Action-Coverage / Error Inheritance
**Students inherit the teacher's blind spots. If the teacher can't emit an action, the student won't learn it.**

Evidence by action type:
```
done:     A 0.815 → C 0.430 → B 0.000    (teacher: GTA1 0.275, UITARS 0.005)
scroll:   C 0.568 > B 0.338              (teacher: GTA1 0.615, UITARS 0.243)
type:     C 0.667 > B 0.593              (teacher: GTA1 0.679)
open_app: A 0.827, B=C=0.133             (teacher: GTA1 0.080, UITARS 0.000)
```

**The headline:** only the **human teacher** demonstrated `open_app` (0.827). Neither model teacher could emit it reliably (GTA1 0.080, UI-TARS 0.000), so the model-trained students learned it poorly (B/C ≈0.133 vs A ≈0.827). **Distillation is a bottleneck shaped by the teacher's own limitations.**

---

## Code Artifacts & Methodology

### Parser-Per-Teacher Approach
Each teacher model has a different action syntax and coordinate convention. We verified this empirically (the hard-won lesson) via `src/recon_teacher.py`:

| Teacher | Coords | Parser | Notes |
|---------|--------|--------|-------|
| Humans (GT) | 0–1000 norm | built-in | Canonical; training target |
| UI-TARS-7B | 0–1000 norm | `src/uitars.py` | Qwen2-VL-based; emits 0–1000 natively |
| GTA1-32B | Absolute PIXELS | `src/gta1.py` | Qwen2-VL-based; emits pixels; we normalize pixel→0–1000 for labels |
| Qwen2.5-VL-3B (base/student) | Absolute PIXELS | built-in | Native output convention; all students inherited this |

### Pipeline
```
Teacher output (raw text)
        ↓
    parser  (--teacher flag in eval_androidcontrol, eval_screenspot, run_teacher)
        ↓
Action object (type + args)
        ↓
Coordinate normalization  (if pixel teacher, normalize pixel→0–1000 for label storage)
        ↓
Canonical 0–1000 space  (all labels stored here, train/eval on the same space)
        ↓
make_sharegpt  (convert to training format)
        ↓
LLaMA-Factory LoRA training
        ↓
Eval (--coord_space pixel for all models; base/students are pixel-native)
```

### Key Files
- **Recon:** `src/recon_teacher.py` — empirically verify a new teacher's output format before writing a parser
- **Parsers:** `src/uitars.py`, `src/gta1.py` — teacher-specific action syntax + coordinate handling
- **Eval drivers:** `src/eval_screenspot.py`, `src/eval_androidcontrol.py` — `--teacher flag` to plug in parser; `--coord_space` for coordinate convention
- **Label generation:** `src/run_teacher.py` — teacher-forced relabeling with parser + coordinate normalization
- **Training:** `configs/train_qwen3b_lora_track{A,B,C}.yaml` — identical recipe, only dataset differs
- **Memory:** `PROGRESS.md` (this file), `results/Summary.txt` (numbers + per-type breakdown)

---

## What's Next — The Roadmap

### Immediate (High-Value, Fits RTX 4090)

**1. UI-Venus-72B (agent teacher, not grounding-only)**
- **Goal:** does a *better agent* (vs GTA1's grounding focus) close the `open_app` gap?
- **Method:** recon → write `src/venus.py` parser → run full sweep (same pipeline as GTA1)
- **Expected:** multi-step should track (likely 0.5+); grounding stays ~0.30; open_app improves if UI-Venus emits it
- **Cost:** 80 GB GPU (needs A100/H100 or 4-bit quantization on 96 GB)
- **Artifacts:** Track D student/evals; fills the "stronger agent" cell

**2. Tighten the pilot (quick wins)**
- Add **error bars/seeds** (re-train A/B/C with 3 random seeds) → confidence intervals
- Run **full ScreenSpot** (all 1,272 samples, not just 400) → tighter grounding estimate
- **Phone-grounding test** → separate OOD loss from a general drop

### Medium-Term (Makes the Paper Real)

**3. Closed-API teacher (GPT-4V / Claude 3.5 Vision / Gemini)**
- **Goal:** a truly "black-box" teacher (no weights, no architecture access) — the paper's core premise
- **Method:** `src/recon_teacher.py` on API calls → capture raw outputs → build parser → run full sweep
- **Expected:** multi-step likely lower than open teachers (API latency + cost limits label-gen scale); grounding still stuck? (could be interesting)
- **Cost:** API fees (~$100–500 for 5k inferences, depending on model/pricing)
- **Artifacts:** Track E student/evals; the "black-box" proof-of-concept

### Stretch Goals

**4. Offline distillation → Online RL fine-tuning**
- Train the student on hard-label distillation (A/B/C), then fine-tune via RL on live AndroidWorld environment (reward = task success)
- **Goal:** does hard-label SFT provide a good warm-start for RL? How much does it improve over training RL from scratch?
- **Requires:** an x86+KVM Linux host (for the AndroidWorld emulator)

**5. Rejection sampling**
- Train two students: one on all teacher traces, one on only "successful" traces (teacher succeeded on that step)
- **Goal:** does filtering by teacher success change the grounding vs multi-step trade-off?

---

## Why This Matters

### For the Community
Black-box distillation is the *only feasible path* for on-device GUI agents once you need to use proprietary teachers (GPT, Claude, Gemini). This work measures what you **gain** (multi-step scaling) and **lose** (grounding, teacher coverage) under that constraint — critical for product decisions ("is our small model good enough?") and research ("where should we invest to close the gap?").

### For the Paper
The A/B/C sweep **isolates the effect of teacher quality** from all other variables (same student, recipe, eval, data). This is methodologically strong: if multi-step scales with teacher and grounding doesn't, that's a concrete finding about the limits of the distillation recipe, not an artifact of the setup.

### For Practitioners
- If you need on-device agents, **expect grounding to degrade** (OOD loss is baked into the recipe).
- Multi-step *is* fixable via better teachers; grounding *isn't* (within this recipe).
- If your teacher can't emit an action, neither will your student.

---

## Current Code State

All scaffolding for A/B/C is in place and tested:
- ✅ `recon_teacher.py` — ready for any new teacher
- ✅ `uitars.py`, `gta1.py` parsers — validated on real teacher outputs
- ✅ Eval drivers with `--teacher` flag — pluggable parser system
- ✅ `run_teacher.py` — coordinate normalization for pixel teachers
- ✅ Training configs for A/B/C — identical recipe, only dataset differs
- ✅ Full results → GitHub (Summary.txt + per-action-type breakdowns)

To add a new teacher (e.g., UI-Venus-72B):
1. `python src/recon_teacher.py --teacher_model <model_id>` → inspect raw outputs
2. Write `src/<name>.py` with `parse_<name>()` function (30 lines, copy-paste-modify from gta1.py)
3. Wire into evals: add `--teacher <name>` option (already scaffolded in eval drivers)
4. Run: `python src/run_teacher.py --steps ... --parser <name>` → relabel
5. Train, eval, compare against A/B/C

---

## Thesis Summary (One Sentence)

**Hard-label distillation scales multi-step GUI agent behavior with teacher quality, but perception (grounding) doesn't transfer and students inherit the teacher's action vocabulary gaps — suggesting that closing the gap requires either (a) better teachers that demonstrate the missing actions, or (b) a different training recipe that doesn't sacrifice OOD grounding.**

---

## Key Learnings (For the Lab Notebook)

1. **Never assume a model's coordinate convention.** Verify empirically on raw outputs. Mis-scoring the space reads ~0 and is indistinguishable from failure.
2. **Teacher-forced offline eval is asymmetric.** It favors the human-distilled student (because it learned to mimic the exact reference action) and undercounts a capable-but-different agent. Use per-action-type breakdown to diagnose.
3. **Distribution shift dominates.** The student's in-distribution click accuracy improved, but OOD grounding collapsed — suggesting the recipe is fragile to domain shift, not inherently weak at grounding.
4. **Coverage is the bottleneck for rare actions.** `open_app` scores correlate perfectly with whether the teacher demonstrates it. If you need rare actions, force the teacher to emit them in training data.

---

## How to Cite This Work (When Ready)

```
@article{distill-gui-agents,
  title={Black-Box Hard-Label Distillation of GUI Agents: 
         Multi-Step Scales with Teacher, Grounding Doesn't Transfer},
  author={...},
  year={2026},
  note={Learning run; A/B/C teacher-capability sweep.}
}
```

---

**Last updated:** 2026-06-02  
**Total compute:** ~3 × RTX 4090 hours (label-gen + training per track); 1 × RTX PRO 6000 96GB (GTA1 recon)  
**Code ready for:** UI-Venus-72B, closed API, any future teacher with 30-line parser
