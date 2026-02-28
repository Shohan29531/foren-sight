from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import open_clip
from PIL import Image


@dataclass
class ClipConfig:
    model_name: str
    pretrained: str
    device: str


class ClipEmbedder:
    def __init__(self, cfg: ClipConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

    @torch.no_grad()
    def embed_images(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            imgs = []
            for p in batch:
                img = Image.open(p).convert("RGB")
                imgs.append(self.preprocess(img))
            x = torch.stack(imgs).to(self.device)
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy().astype(np.float32))
        return np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(texts)
        tokens = tokens.to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def default_embedder() -> ClipEmbedder:
    model_name = os.getenv("CLIP_MODEL", "ViT-B-32")
    pretrained = os.getenv("CLIP_PRETRAINED", "openai")
    device = os.getenv("CLIP_DEVICE", "cpu")
    return ClipEmbedder(ClipConfig(model_name=model_name, pretrained=pretrained, device=device))
