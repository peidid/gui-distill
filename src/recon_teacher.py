"""Recon a teacher model's RAW output before writing a parser.

The hard-won lesson of this project: never assume a model's action syntax or
coordinate convention — print a few raw outputs and verify empirically. Run this
for any new teacher (Track C: UI-Venus / GTA1; later: a closed API), look at the
coordinate magnitudes vs the image size, then write the parser (see src/uitars.py
for the UI-TARS one).

    python src/recon_teacher.py --teacher_model inclusionAI/UI-Venus-Navi-72B
    python src/recon_teacher.py --teacher_model HelloKKMe/GTA1-32B
"""

import argparse
import sys
import tempfile

sys.path.insert(0, "src")
from prompt import SYSTEM, build_prompt, build_prompt_parts
from schema import load_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--steps", default="data/androidcontrol/steps_test.jsonl")
    ap.add_argument("--n", type=int, default=6, help="samples per benchmark")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", default="results/recon.txt")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    from qwen_vl_utils import process_vision_info

    print("loading", args.teacher_model, "(this can be slow for 32B/72B)", flush=True)
    proc = AutoProcessor.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoVLM.from_pretrained(args.teacher_model, torch_dtype="auto",
                                    device_map="auto", trust_remote_code=True)

    def gen(image_path, user):
        user = user.replace("<image>", "").lstrip("\n")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image", "image": image_path},
                                             {"type": "text", "text": user}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, videos=vids, padding=True,
                   return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        return proc.batch_decode(g[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

    lines = []
    for s in load_steps(args.steps)[:args.n]:
        raw = gen(s.image_path, build_prompt(s))
        lines.append(f"[AC] img={s.image_w}x{s.image_h} gt={s.action.serialize()}\n     RAW: {raw!r}")
        print(lines[-1], flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test").select(range(args.n))
        for x in ds:
            img = x["image"]
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False); img.save(f.name)
            raw = gen(f.name, build_prompt_parts(f"Click on: {x['instruction']}"))
            lines.append(f"[SS] img={img.width}x{img.height} instr={x['instruction']!r}\n     RAW: {raw!r}")
            print(lines[-1], flush=True)
    except Exception as e:
        print("ScreenSpot recon skipped:", e, flush=True)

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines))
    print("\nsaved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
