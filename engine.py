"""Qwen vision model loading and inference for Enzino Video2Prompt.

Qwen vision models already on disk (models/VLM, models/LLM, the Hugging Face
cache...) are listed first. Missing ones are downloaded to ComfyUI/models/VLM
the first time they are used.
"""

import gc
import json
import os

import numpy as np
import torch

try:
    import folder_paths
    import comfy.model_management as mm
except ImportError:  # running outside ComfyUI (tests)
    folder_paths = None
    mm = None

# Downloadable models: short name -> (hf repo, folder name, approx bf16 weight size in GB)
MODELS = {
    "Qwen3-VL-4B": ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct", 8.9),
    "Qwen3-VL-8B": ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct", 17.5),
    "Qwen3-VL-2B": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct", 4.3),
    "Qwen3.5-4B": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", 9.3),
    "Qwen3.5-9B": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B", 19.3),
}
RECOMMENDED = "Qwen3-VL-4B"
# Labels used by v1.0.0 workflows
LEGACY = {
    "Qwen3-VL-4B (recommended)": "Qwen3-VL-4B",
    "Qwen3-VL-8B (best, 16 GB+)": "Qwen3-VL-8B",
    "Qwen3-VL-2B (low VRAM)": "Qwen3-VL-2B",
}
PRECISIONS = ["auto", "bf16", "8-bit", "4-bit"]

INSTALLED = "\u2713 "   # ✓ already on disk
DOWNLOAD = "\u2193 "    # ↓ will be downloaded

# Qwen vision-language model types transformers can load with AutoModelForImageTextToText
VISION_TYPES = {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}
# Folders other nodes use for the same models (ComfyUI-QwenVL, prompt generators...)
SCAN_SUBDIRS = ["VLM", "LLM", "LLM/Qwen-VL", "Qwen-VL", "qwen", "prompt_generator", "VLM/Qwen"]


def models_root():
    return folder_paths.models_dir if folder_paths else os.environ.get("ENZINO_VLM_DIR", "models")


def vlm_dir():
    path = os.path.join(models_root(), "VLM")
    os.makedirs(path, exist_ok=True)
    return path


def is_complete(path):
    """config.json + every weight shard present (a cut download is not 'installed')."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                shards = set(json.load(f)["weight_map"].values())
            return all(os.path.isfile(os.path.join(path, s)) for s in shards)
        except Exception:  # noqa: BLE001
            return False
    return any(n.endswith(".safetensors") for n in os.listdir(path))


def _is_qwen_vision(path):
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001
        return False
    return cfg.get("model_type") in VISION_TYPES or (
        "vision_config" in cfg and "qwen" in str(cfg.get("model_type", "")).lower())


def _weights_gb(path):
    try:
        return sum(os.path.getsize(os.path.join(path, n)) for n in os.listdir(path) if n.endswith(".safetensors")) / 1e9
    except OSError:
        return None


def find_installed():
    """{display name: folder path} for every complete Qwen vision model found on disk."""
    found = {}
    roots = [os.path.join(models_root(), d) for d in SCAN_SUBDIRS]
    if folder_paths:
        for key in ("VLM", "LLM"):
            try:
                roots += folder_paths.get_folder_paths(key)
            except Exception:  # noqa: BLE001
                pass
    seen = set()
    for root in roots:
        root = os.path.normpath(root)
        if root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            full = os.path.join(root, name)
            if os.path.isdir(full) and name not in found and is_complete(full) and _is_qwen_vision(full):
                found[name] = full
    # Hugging Face cache (models downloaded by other tools)
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type != "model" or not repo.repo_id.lower().startswith("qwen/"):
                continue
            name = repo.repo_id.split("/", 1)[1]
            for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                snap = str(rev.snapshot_path)
                if name not in found and is_complete(snap) and _is_qwen_vision(snap):
                    found[name] = snap
                    break
    except Exception:  # noqa: BLE001 - no cache / old huggingface_hub
        pass
    return found


def model_choices():
    """Models already on disk first (\u2713), then the ones that can be downloaded (\u2193)."""
    installed = find_installed()
    choices = [INSTALLED + n for n in installed]
    folders = {os.path.basename(os.path.normpath(p)) for p in installed.values()} | set(installed)
    for short, (_repo, folder, size) in MODELS.items():
        if folder not in folders:
            choices.append(f"{DOWNLOAD}{short} \u00b7 {size:g} GB")
    return choices


def default_choice(choices):
    rec_folder = MODELS[RECOMMENDED][1]
    for c in choices:  # recommended model, installed
        if c == INSTALLED + rec_folder:
            return c
    for c in choices:  # any installed model
        if c.startswith(INSTALLED):
            return c
    for c in choices:
        if c.startswith(DOWNLOAD + RECOMMENDED + " "):
            return c
    return choices[0]


def _strip(label):
    for pre in (INSTALLED, DOWNLOAD):
        if label.startswith(pre):
            label = label[len(pre):]
    return label.split(" \u00b7 ")[0].strip()


def resolve_model(label):
    """Return (local_path, approx_size_gb). Downloads if needed."""
    name = _strip(LEGACY.get(label, label))
    name = LEGACY.get(name, name)
    installed = find_installed()
    short = name if name in MODELS else next((k for k, v in MODELS.items() if v[1] == name), None)
    if name in installed:
        return installed[name], _weights_gb(installed[name])
    if short and MODELS[short][1] in installed:
        p = installed[MODELS[short][1]]
        return p, _weights_gb(p)
    if short is None:
        raise RuntimeError(f"Model not found on disk: {name}. Click Refresh (R) or pick another model.")

    repo, folder, size = MODELS[short]
    local = os.path.join(vlm_dir(), folder)
    print(f"[Enzino Video2Prompt] Downloading {repo} (~{size} GB) to {local} \u2014 first use only.")
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    # Small files first (config, tokenizer...), then the weights one by one so the
    # console shows a real GB progress bar instead of a "10/12 files" counter that seems stuck.
    snapshot_download(repo_id=repo, local_dir=local, allow_patterns=["*.json", "*.txt", "*.jinja", "*.model", "*.py"])
    shards = sorted(f for f in list_repo_files(repo) if f.endswith(".safetensors"))
    for i, fname in enumerate(shards, 1):
        print(f"[Enzino Video2Prompt] Weights {i}/{len(shards)}: {fname}")
        hf_hub_download(repo_id=repo, filename=fname, local_dir=local)
    print("[Enzino Video2Prompt] Download complete.")
    return local, size


def _device():
    if mm is not None:
        return mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_vram_gb(device):
    if device.type != "cuda":
        return 0.0
    if mm is not None:
        return mm.get_free_memory(device) / 1024**3
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024**3


class _Cache:
    key = None
    model = None
    processor = None


def unload():
    _Cache.key = _Cache.model = _Cache.processor = None
    gc.collect()
    if mm is not None:
        mm.soft_empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(label, precision="auto"):
    if _Cache.key == (label, precision) and _Cache.model is not None:
        return _Cache.model, _Cache.processor

    unload()
    if mm is not None:
        # Free the GPU from diffusion models before loading the VLM.
        mm.unload_all_models()
        mm.soft_empty_cache()

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise RuntimeError(
            "Your 'transformers' is too old for Qwen3-VL. Update it in the ComfyUI python: "
            "python -m pip install -U \"transformers>=4.57\""
        ) from e

    path, size = resolve_model(label)
    device = _device()

    prec = precision
    if prec == "auto":
        if device.type != "cuda":
            prec = "bf16"
        else:
            need = (size or 9.0) * 1.15 + 1.5  # weights + vision activations
            prec = "bf16" if _free_vram_gb(device) >= need else "4-bit"

    if device.type == "cuda":
        target = {"": device.index if device.index is not None else 0}
    else:
        target = {"": "cpu"}
        if prec in ("8-bit", "4-bit"):
            prec = "bf16"
    kwargs = {"low_cpu_mem_usage": True}
    if prec in ("8-bit", "4-bit"):
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("[Enzino Video2Prompt] bitsandbytes not installed: loading in bf16 with CPU offload instead.")
            prec = "bf16-offload"
        else:
            if prec == "4-bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = target

    if prec in ("bf16", "bf16-offload"):
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto" if prec == "bf16-offload" else target

    try:
        model = AutoModelForImageTextToText.from_pretrained(path, attn_implementation="sdpa", **kwargs)
    except TypeError:
        # transformers < 4.56 used torch_dtype
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(path, attn_implementation="sdpa", **kwargs)
    except ValueError as e:
        if "qwen3" in str(e).lower() or "does not recognize" in str(e):
            raise RuntimeError(
                "This model needs a newer 'transformers'. In the ComfyUI python: "
                "python -m pip install -U transformers"
            ) from e
        raise
    model.eval()
    processor = AutoProcessor.from_pretrained(path)
    print(f"[Enzino Video2Prompt] Loaded {label} ({prec}).")
    _Cache.key, _Cache.model, _Cache.processor = (label, precision), model, processor
    return model, processor


def _chat_text(processor, messages):
    template = getattr(processor, "chat_template", None) or ""
    extra = {"enable_thinking": False} if "enable_thinking" in str(template) else {}
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _inputs_as_video(processor, system, user, frames, info):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user}]},
    ]
    text = _chat_text(processor, messages)
    video = np.stack(frames)  # T,H,W,C uint8
    meta = {
        "fps": info["fps"],
        "total_num_frames": info["total_frames"],
        "duration": info["duration"],
        "frames_indices": info["indices"],
        "width": info["width"],
        "height": info["height"],
    }
    try:
        return processor(text=[text], videos=[video], video_metadata=[meta], do_sample_frames=False,
                         return_tensors="pt")
    except TypeError:
        return processor(text=[text], videos=[video], return_tensors="pt")


def _inputs_as_images(processor, system, user, frames, info):
    from PIL import Image

    content = []
    labels = info.get("labels") or [f"{idx / info['fps']:.1f} s" for idx in info["indices"]]
    for k, (f, lab) in enumerate(zip(frames, labels), 1):
        content.append({"type": "text", "text": f"Frame {k} ({lab}):"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": user})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
    text = _chat_text(processor, messages)
    return processor(text=[text], images=[Image.fromarray(f) for f in frames], return_tensors="pt")


@torch.inference_mode()
def generate(model, processor, system, user, frames, info, max_new_tokens, as_video=True):
    # Several shots: send labelled stills so the model knows which frames belong to which shot.
    if info.get("multi_shot"):
        as_video = False
    builders = [_inputs_as_video, _inputs_as_images] if as_video else [_inputs_as_images]
    last_err = None
    for build in builders:
        try:
            inputs = build(processor, system, user, frames, info)
            break
        except Exception as e:  # noqa: BLE001 - video path unsupported by this processor
            last_err = e
            print(f"[Enzino Video2Prompt] {build.__name__} failed ({e}); trying next mode.")
    else:
        raise RuntimeError(f"The processor refused the frames: {last_err}")

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    dtype = getattr(model, "dtype", None)
    for k in ("pixel_values", "pixel_values_videos"):
        if k in inputs and dtype is not None and dtype in (torch.bfloat16, torch.float16, torch.float32):
            inputs[k] = inputs[k].to(dtype)

    out = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        repetition_penalty=1.05,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
