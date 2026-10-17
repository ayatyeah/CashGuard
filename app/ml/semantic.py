"""CLIP ViT-B/32 (MIT) image/text embeddings: zero-shot scene-state features and inputs for trained linear probes."""
from __future__ import annotations

import os

import cv2
import numpy as np

CLIP_ID = os.environ.get("FORTE_CLIP", "openai/clip-vit-base-patch32")

# zero-shot probes: (name, positive prompts, negative prompts)
PROBES = {
    "desk_items": (["banknotes lying on an office desk", "a stack of money on a desk", "a green cash bag on a desk",
                    "a notebook and stamps left on a desk", "papers and money on a desk"],
                   ["a tidy office desk with only a computer and a phone", "an empty office desk"]),
    "screen_on": (["a laptop with a bright screen turned on", "a computer screen showing a spreadsheet"],
                  ["a laptop with a black screen", "a closed laptop"]),
    "drinking": (["a woman drinking coffee from a cup", "a person holding a coffee cup", "a takeaway coffee cup on a desk"],
                 ["a woman working at a desk", "a woman counting banknotes"]),
    "phone": (["a woman looking at her smartphone", "a person holding a mobile phone in hands"],
              ["a woman working at a desk", "a woman counting banknotes"]),
    "bag_clothes": (["a handbag and a jacket on an office desk", "a coat lying on a desk", "a purse on a desk"],
                    ["an office desk with a computer and a banknote counter"]),
    "cash_bag": (["a person carrying a green bag", "a green collection bag"], ["a person in an office"]),
    "dark": (["a dark room with lights off"], ["a brightly lit office"]),
}


class ClipEmbedder:
    def __init__(self, threads: int = 6):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = CLIPModel.from_pretrained(CLIP_ID).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self._text = None

    def images(self, bgr_list: list[np.ndarray]) -> np.ndarray:
        if not bgr_list:
            return np.zeros((0, 512), np.float32)
        rgb = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr_list]
        with self.torch.no_grad():
            px = self.proc(images=rgb, return_tensors="pt")["pixel_values"]
            out = self.model.get_image_features(pixel_values=px)
            e = out if isinstance(out, self.torch.Tensor) else out.pooler_output
        e = e.numpy().astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            tok = self.proc(text=texts, return_tensors="pt", padding=True)
            out = self.model.get_text_features(**tok)
            e = out if isinstance(out, self.torch.Tensor) else out.pooler_output
        e = e.numpy().astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def probe_bank(self):
        if self._text is None:
            bank = {}
            for k, (pos, neg) in PROBES.items():
                bank[k] = (self.texts(pos), self.texts(neg))
            self._text = bank
        return self._text


def probe_scores(emb: np.ndarray, bank: dict) -> dict:
    """emb (N,512) normalized -> {probe: (N,) score = softmax-prob mass of positive prompts}."""
    out = {}
    for k, (p, n) in bank.items():
        logits = 100.0 * emb @ np.concatenate([p, n]).T
        logits -= logits.max(1, keepdims=True)
        pr = np.exp(logits)
        pr /= pr.sum(1, keepdims=True)
        out[k] = pr[:, : len(p)].sum(1)
    return out


def crop(img: np.ndarray, box) -> np.ndarray:
    if box is None:
        return img
    x1, y1, x2, y2 = [int(v) for v in box]
    c = img[max(0, y1):y2, max(0, x1):x2]
    return c if c.size else img
