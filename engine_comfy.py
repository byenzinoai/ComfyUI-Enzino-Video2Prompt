"""Second engine: the Qwen3-VL files from ComfyUI/models/text_encoders.

Krea 2, Fibo 2, Qwen-Image 2.1... ship full Qwen3-VL models (vision tower included)
as single .safetensors text encoders. ComfyUI can run them as a chat model
(see its "Generate Text" node), so we reuse them instead of downloading anything.
VRAM is handled by ComfyUI's own model management.
"""

import os

import numpy as np
import torch

import folder_paths
import comfy.sd

TE_TAG = " · text encoder"
_VISION_KEY = "model.visual.deepstack_merger_list.0.norm.weight"
_MERGER_KEY = "model.visual.merger.linear_fc2.weight"
_SIZES = {2048: "2B", 2560: "4B"}

_scan_cache = {}  # path -> (mtime, size_label or None)


def _inspect(path):
    """Read only the safetensors header: is it a Qwen3-VL that can generate text?"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _scan_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    label = None
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if _VISION_KEY in keys and _MERGER_KEY in keys:
                width = f.get_slice(_MERGER_KEY).get_shape()[0]
                size = _SIZES.get(width, "8B")
                # 2B/4B tie the output head to the embeddings; 8B needs its own lm_head to write text.
                if size != "8B" or "lm_head.weight" in keys:
                    label = size
    except Exception:  # noqa: BLE001 - gguf, broken file, other format
        label = None
    _scan_cache[path] = (mtime, label)
    return label


def find_text_encoders():
    """{choice label: file path} for usable Qwen3-VL text encoder files."""
    out = {}
    try:
        names = folder_paths.get_filename_list("text_encoders")
    except Exception:  # noqa: BLE001
        return out
    for name in names:
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("text_encoders", name)
        if path and _inspect(path):
            out[f"✓ {name}{TE_TAG}"] = path
    return out


def is_te_label(label):
    return label.endswith(TE_TAG)


class _Cache:
    path = None
    clip = None


def unload():
    _Cache.path = _Cache.clip = None


def load(label):
    path = find_text_encoders().get(label)
    if path is None:
        name = label[2:-len(TE_TAG)] if is_te_label(label) else label
        raise RuntimeError(f"Text encoder not found or not usable for text generation: {name}")
    if _Cache.path == path and _Cache.clip is not None:
        return _Cache.clip
    unload()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,  # generic Qwen3-VL chat path
    )
    print(f"[Enzino Video2Prompt] Using text encoder {os.path.basename(path)} ({_inspect(path)}).")
    _Cache.path, _Cache.clip = path, clip
    return clip


def generate(clip, system, user, frames, info, max_new_tokens):
    images = [torch.from_numpy(np.ascontiguousarray(f)).float().div(255.0).unsqueeze(0) for f in frames]
    labels = info.get("labels") or [f"{i / info['fps']:.1f} s" for i in info["indices"]]
    # Our own chat text, so each frame gets its label right before it (ComfyUI fills the image slots in order).
    vision = "".join(f"Frame {k} ({lab}): <|vision_start|><|image_pad|><|vision_end|>\n" for k, lab in enumerate(labels, 1))
    chat = (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\nFrames of ONE video clip, in order:\n{vision}\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    tokens = clip.tokenize(chat, images=images, skip_template=True, min_length=1, thinking=False)
    ids = clip.generate(
        tokens,
        do_sample=False,
        max_length=int(max_new_tokens),
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.05,
        seed=0,
    )
    return clip.decode(ids)
