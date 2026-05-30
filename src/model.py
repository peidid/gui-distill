"""Thin inference wrapper around the student/teacher VLM.

Backends (pick with --backend):
  dummy : returns a canned action. No deps. Use to smoke-test the eval pipeline.
  mlx   : Apple-Silicon native (mlx-vlm). Runs Qwen2.5-VL-3B on your Mac for
          BASELINE numbers before you rent a GPU. Slow but free.
  hf    : transformers + (optional) PEFT LoRA adapter. This is what you run on the
          rented GPU box to evaluate the trained student.

All backends expose: generate(image_path, system, user) -> raw_text.
The `user` text may contain the literal "<image>" token (as built by prompt.py);
backends that pass the image structurally strip it so it isn't duplicated.
"""

from prompt import IMAGE_TOKEN


def _strip_image_token(user):
    return user.replace(IMAGE_TOKEN, "").lstrip("\n")


class DummyModel:
    """Returns a fixed action — only for testing that the eval harness runs."""
    def __init__(self, canned="Thought: testing.\nAction: click(500, 500)"):
        self.canned = canned

    def generate(self, image_path, system, user):
        return self.canned


class MLXModel:
    """mlx-vlm backend (Apple Silicon). `pip install mlx-vlm`.

    Note: mlx-vlm's generate()/apply_chat_template() signatures drift between
    versions. If you hit a TypeError, check `python -c "import mlx_vlm; print(mlx_vlm.__version__)"`
    and adjust the two flagged lines.
    """
    def __init__(self, model_path="mlx-community/Qwen2.5-VL-3B-Instruct-4bit", max_tokens=128):
        from mlx_vlm import load
        self.gen_max = max_tokens
        self.model, self.processor = load(model_path)
        self.config = self.model.config

    def generate(self, image_path, system, user):
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _strip_image_token(user)},
        ]
        prompt = apply_chat_template(self.processor, self.config, messages, num_images=1)  # (1)
        out = mlx_generate(self.model, self.processor, prompt, image=[image_path],
                           max_tokens=self.gen_max, verbose=False)              # (2)
        return out.text if hasattr(out, "text") else str(out)


class HFModel:
    """transformers backend. Loads Qwen2.5-VL and, optionally, a LoRA adapter."""
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct", adapter=None,
                 max_tokens=128):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except ImportError:  # older transformers naming
            from transformers import Qwen2VLForConditionalGeneration as VLModel
        self.torch = torch
        self.gen_max = max_tokens
        self.model = VLModel.from_pretrained(model_path, torch_dtype="auto",
                                             device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, image_path, system, user):
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": _strip_image_token(user)},
            ]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.gen_max,
                                      do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def load_model(backend="dummy", model_path=None, adapter=None, max_tokens=128):
    if backend == "dummy":
        return DummyModel()
    if backend == "mlx":
        kw = {"max_tokens": max_tokens}
        if model_path:
            kw["model_path"] = model_path
        return MLXModel(**kw)
    if backend == "hf":
        kw = {"adapter": adapter, "max_tokens": max_tokens}
        if model_path:
            kw["model_path"] = model_path
        return HFModel(**kw)
    raise ValueError(f"unknown backend: {backend}")


if __name__ == "__main__":
    m = load_model("dummy")
    print("dummy ->", repr(m.generate("x.png", "sys", "<image>\nuser")))
    assert "Action:" in m.generate("x.png", "s", "u")
    print("model.py: dummy backend ok")
