"""Local VLM (Qwen3-VL-2B-Instruct, Apache-2.0) used as a *feature extractor*: it answers short yes/no questions
about key moments proposed by the CV pipeline. The answers become features for the trained scorer.

Backends: transformers on CPU (Linux dev box) and mlx-vlm on Apple Silicon (FORTE_VLM_BACKEND=mlx).
No network access at inference time (weights from the local cache).
"""
from __future__ import annotations

import json
import os
import re
import time

import cv2
import numpy as np

VLM_ID = os.environ.get("FORTE_VLM", "Qwen/Qwen3-VL-2B-Instruct")
MLX_ID = os.environ.get("FORTE_VLM_MLX", "mlx-community/Qwen3-VL-2B-Instruct-4bit")
MAX_SIDE = int(os.environ.get("FORTE_VLM_MAX_SIDE", "640"))

SYSTEM = ("You are a careful assistant that inspects CCTV frames from a bank cash office. "
          "Answer only with a compact JSON object, no explanations. If unsure, answer false.")


def compose(images: list[np.ndarray], max_side: int = MAX_SIDE) -> np.ndarray:
    """Concatenate frames horizontally (same height) and downscale so the longest side <= max_side."""
    h = min(im.shape[0] for im in images)
    row = np.hstack([cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images])
    s = max_side / max(row.shape[:2])
    if s < 1:
        row = cv2.resize(row, (int(row.shape[1] * s), int(row.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return row


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    raw = m.group(0)
    pyfix = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", raw))
    for cand in (raw, raw.replace("'", '"'), pyfix, pyfix.replace("'", '"')):
        try:
            d = json.loads(cand)
            return {k: (v if not isinstance(v, str) else v.strip().lower() in ("true", "yes", "1")) for k, v in d.items()}
        except Exception:
            continue
    out = {}
    for k, v in re.findall(r'["\']?(\w+)["\']?\s*:\s*(true|false|True|False|yes|no)', raw):
        out[k] = v.lower() in ("true", "yes")
    return out


class VLM:
    def __init__(self, threads: int | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.backend = os.environ.get("FORTE_VLM_BACKEND", "transformers")
        self.calls, self.seconds = 0, 0.0
        if self.backend == "mlx":
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            self.model, self.proc = load(MLX_ID)
            self._tmpl = apply_chat_template
        else:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            torch.set_num_threads(threads or max(2, (os.cpu_count() or 4) - 2))
            self.torch = torch
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(VLM_ID, dtype=torch.bfloat16).eval()
            self.proc = AutoProcessor.from_pretrained(VLM_ID)

    def ask(self, images: list[np.ndarray], question: str, max_new_tokens: int = 80) -> str:
        img = compose(images)
        t0 = time.time()
        if self.backend == "mlx":
            from mlx_vlm import generate
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
            prompt = self._tmpl(self.proc, self.model.config, msgs, num_images=1)
            out = generate(self.model, self.proc, prompt, [pil], max_tokens=max_new_tokens, temperature=0.0,
                           verbose=False)
            text = out if isinstance(out, str) else getattr(out, "text", str(out))
        else:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": question}]}]
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt")
            with self.torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self.proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.calls += 1
        self.seconds += time.time() - t0
        return text

    def ask_json(self, images, question, max_new_tokens: int = 80) -> dict:
        return parse_json(self.ask(images, question, max_new_tokens))

    def ask_flags(self, images, q) -> dict:
        keys, prompt = q
        a = self.ask_json(images, prompt, max_new_tokens=90)
        return {k: bool(a[k]) for k in keys if k in a}


# ---- question bank: (feature keys, prompt asking for a JSON object with exactly these boolean keys)
def _json_prompt(context: str, keys: list[str]) -> str:
    return f"{context} Answer JSON with booleans: {{" + ", ".join(f'"{k}": ...' for k in keys) + "}"


_T = ["teller_holds_phone", "teller_drinks", "cup_or_food_on_desk", "handbag_or_coat_on_desk", "cash_on_desk"]
_V = ["client_shows_id_card", "client_gives_cash", "client_gives_bag", "teller_uses_counting_machine",
      "cash_stacks_mixed_on_desk"]
_D = ["cash_left_on_desk", "documents_or_notebook_left", "bag_left_on_desk", "laptop_screen_on", "desk_is_tidy"]
Q_TELLER = (_T, _json_prompt("The image shows a bank teller at her desk (several moments).", _T))
Q_VISITOR = (_V, _json_prompt("The image shows moments of a client visit at a bank teller desk, in time order "
                              "from left to right.", _V))
Q_DESK = (_D, _json_prompt("The image shows a bank teller desk after the teller walked away.", _D))
_S = ["safe_door_open", "key_in_safe_lock", "cash_visible_in_safe"]
Q_SAFE = (_S, _json_prompt("The image shows a metal safe (cash cabinet) in a bank office, several moments.", _S))
