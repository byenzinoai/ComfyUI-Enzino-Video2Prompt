"""Enzino Video2Prompt: a video in, a MiniMax H3 prompt out."""

import os
import time

import numpy as np
import torch

import folder_paths

from . import engine, engine_comfy, plan, prompts, video_io

NONE = "(none)"
FRAME_COUNTS = ["8", "12", "16", "24", "32"]
DETAILS = {"Fast (384 px)": 384, "Balanced (512 px)": 512, "Fine (768 px)": 768}


def _input_videos():
    in_dir = folder_paths.get_input_directory()
    files = []
    for root, _dirs, names in os.walk(in_dir):
        for n in names:
            if n.lower().endswith(video_io.VIDEO_EXTENSIONS):
                files.append(os.path.relpath(os.path.join(root, n), in_dir).replace("\\", "/"))
    return [NONE] + sorted(files)


def _frames_to_tensor(frames):
    h, w = frames[0].shape[:2]
    same = [f if f.shape[:2] == (h, w) else video_io.resize_max_side(f, max(h, w)) for f in frames]
    arr = np.stack([f[:h, :w] for f in same]).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _save_thumbs(frames, max_side=192):
    """Small previews for the node card (temp folder, like Preview Image)."""
    from PIL import Image

    out_dir = folder_paths.get_temp_directory()
    os.makedirs(out_dir, exist_ok=True)
    stamp = f"enzino_v2p_{int(time.time() * 1000)}"
    results = []
    for i, f in enumerate(frames):
        name = f"{stamp}_{i:02d}.jpg"
        Image.fromarray(video_io.resize_max_side(f, max_side)).save(os.path.join(out_dir, name), quality=82)
        results.append({"filename": name, "subfolder": "", "type": "temp"})
    return results


class EnzinoVideo2Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        hf = engine.model_choices()
        te = list(engine_comfy.find_text_encoders())
        # on disk first (downloaded models, then text encoders), then downloadable ones
        choices = [c for c in hf if c.startswith(engine.INSTALLED)] + te + [c for c in hf if not c.startswith(engine.INSTALLED)]
        return {
            "required": {
                "video": (_input_videos(), {"video_upload": True, "tooltip": "Video from ComfyUI/input. Ignored if a VIDEO or IMAGE input is connected."}),
                "model": (choices, {"default": engine.default_choice(choices), "tooltip": "\u2713 = already on your disk (models/VLM, models/LLM, Hugging Face cache, or a Qwen3-VL file in text_encoders). \u2193 = downloaded to ComfyUI/models/VLM on first use."}),
                "format": (prompts.FORMATS, {"default": "H3 official", "tooltip": "H3 official: MiniMax's own prompt structure (integrated_multimodal_description / overall_soundscape / non_diegetic_music, [Shot N] At 00:03.500 cuts). Simple paragraph: one plain paragraph."}),
                "mode": (prompts.MODES, {"default": "Faithful", "tooltip": "Faithful: recreate the whole shot (T2V). Motion only: actions and camera only, for animating your own image (I2V)."}),
                "length": (prompts.LENGTHS, {"default": "Detailed"}),
                "long_clip": (plan.LONG_MODES, {"default": "Fit into 15 s", "tooltip": "H3 makes 4-15 s videos. Longer clips: the whole clip squeezed into one 15 s prompt (cut times scaled to fit), or only the window below."}),
                "window_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1, "tooltip": "Window mode: start, in seconds."}),
                "window_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1, "tooltip": "Window mode: end, in seconds (0 = start + 15 s). Max 15 s long."}),
                "frames": (FRAME_COUNTS, {"default": "16", "tooltip": "Frames read per prompt, spread evenly. More = better timing, slower."}),
                "detail": (list(DETAILS.keys()), {"default": "Fine (768 px)", "tooltip": "Size of the frames sent to the model."}),
                "sound": ("BOOLEAN", {"default": True, "label_on": "describe sound", "label_off": "silent", "tooltip": "H3 generates sound: describe the likely ambience (inferred from the picture). Off = silent video."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "label_on": "keep in VRAM", "label_off": "free VRAM after", "tooltip": "Keep the model in VRAM between runs (faster, but blocks VRAM for H3)."}),
                "notes": ("STRING", {"multiline": True, "default": "", "placeholder": "Optional changes, e.g. replace the man with a woman in a red coat, night instead of day"}),
            },
            "optional": {
                "video_in": ("VIDEO", {"tooltip": "From the core Load Video node."}),
                "images": ("IMAGE", {"tooltip": "Frames from any video loader (assumed 24 fps)."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "FLOAT", "IMAGE")
    RETURN_NAMES = ("prompt", "first_frame", "duration", "frames")
    OUTPUT_TOOLTIPS = (
        "The H3 prompt.",
        "First frame of the analysed clip, for image-to-video.",
        "Length of the video the prompt describes, in seconds (15 max): set your H3 duration to it.",
        "Every frame the model looked at.",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Enzino/H3"
    DESCRIPTION = "Watches a video with a Qwen vision model and writes a MiniMax H3 prompt that recreates it."

    @classmethod
    def VALIDATE_INPUTS(cls, video, model, **_):
        # Any file name is accepted (uploads, sub-folders, or nothing when an input is wired).
        return True

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        if video and video != NONE:
            path = folder_paths.get_annotated_filepath(video)
            if os.path.exists(path):
                return f"{video}:{os.path.getmtime(path)}"
        return ""

    def run(self, video, model, format, mode, length, long_clip, window_start, window_end, frames, detail,
            sound, keep_model_loaded, notes, video_in=None, images=None, precision="auto"):
        count = int(frames)
        h3 = format == "H3 official"
        i2v = mode == "Motion only"

        if video_in is not None:
            clip = video_io.clip_from_video_object(video_in)
        elif images is not None:
            clip = video_io.clip_from_images(images)
        elif video and video != NONE:
            path = folder_paths.get_annotated_filepath(video)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video not found: {video}")
            clip = video_io.clip_from_file(path)
        else:
            raise ValueError("Enzino Video2Prompt: choose or upload a video first.")

        parts, note = plan.plan(clip.total, clip.fps, list(clip.cuts), long_clip, window_start, window_end)

        # Which frames to read: `count` per part + the first frame of each part.
        sample = {i: p.sample(count) for i, p in enumerate(parts)}
        wanted = {p.start for p in parts}
        for idx in sample.values():
            wanted.update(idx)
        pixels = clip.read(sorted(wanted))

        max_side = DETAILS.get(detail, 512)
        system = prompts.SYSTEM_H3 if h3 else prompts.SYSTEM_SIMPLE
        max_tokens = 640 if length == "Detailed" else 420
        if not h3:
            max_tokens -= 150

        use_te = engine_comfy.is_te_label(model)
        t0 = time.time()
        if use_te:
            engine.unload()
            te_clip = engine_comfy.load(model)
        else:
            engine_comfy.unload()
            mdl, proc = engine.load(model, precision)

        results, thumbs_src, sampled_all = [], [], []
        try:
            for i, part in enumerate(parts):
                idx = [j for j in sample[i] if j in pixels]
                raw = [pixels[j] for j in idx]
                small = [video_io.resize_max_side(f, max_side) for f in raw]
                frame_lines = "\n".join(
                    f"- frame {k + 1}: {part.to_target(j):.2f} s, Shot {part.shot_of(j)}" for k, j in enumerate(idx))
                squeeze = ""
                if part.scale < 0.999:
                    squeeze = (f"The reference clip lasts {part.src_duration:.1f} s; the target video lasts "
                               f"{part.duration:.2f} s, so everything happens a bit faster. Keep every event in the same order, "
                               f"and put each change exactly at the shot times given above (frame times are already converted "
                               f"to the target timeline).")
                user = prompts.build_user_prompt(format, mode, length, part, frame_lines, sound, notes, squeeze)
                info = {
                    "fps": part.fps,
                    "total_frames": part.end - part.start,
                    "duration": part.src_duration,
                    "indices": [j - part.start for j in idx],
                    "width": small[0].shape[1],
                    "height": small[0].shape[0],
                    "labels": [f"{part.to_target(j):.2f} s, Shot {part.shot_of(j)}" for j in idx],
                    "multi_shot": len(part.shots()) > 1,
                }
                budget = max_tokens + 40 * (len(part.shots()) - 1)
                if use_te:
                    text = engine_comfy.generate(te_clip, system, user, small, info, budget)
                else:
                    text = engine.generate(mdl, proc, system, user, small, info, budget)
                text = prompts.clean_h3(text, i2v=i2v, sound=sound) if h3 else prompts.clean_simple(text)
                results.append((part, text))
                thumbs_src += raw
                sampled_all += raw
                print(f"[Enzino Video2Prompt] part {i + 1}/{len(parts)} done ({prompts.word_count(text)} words)")
        finally:
            if not keep_model_loaded:
                engine_comfy.unload() if use_te else engine.unload()
        took = time.time() - t0

        part, prompt = results[0]
        shots = len(part.shots())
        label = (f"H3 PROMPT \u00b7 {part.duration:.1f} s video \u00b7 {shots} shot{'s' if shots > 1 else ''}")

        firsts = _frames_to_tensor([pixels[part.start]])
        frames_t = _frames_to_tensor(sampled_all)
        meta_bits = [f"{video_io.fmt_time(clip.duration)} \u00b7 {clip.fps:.0f} fps"]
        if clip.cuts:
            meta_bits.append(f"{len(clip.cuts)} cut{'s' if len(clip.cuts) > 1 else ''} found")
        if note:
            meta_bits.append(note)
        meta_bits.append(f"{took:.0f}s")
        meta_bits.insert(0, f"set H3 to {round(part.duration)} s")
        ui_parts = [{"label": label, "text": prompt, "words": prompts.word_count(prompt)}]
        return {
            "ui": {"text": [prompt], "v2p_meta": [" \u00b7 ".join(meta_bits)], "v2p_parts": ui_parts,
                   "v2p_frames": _save_thumbs(thumbs_src)},
            "result": (prompt, firsts, float(round(part.duration, 2)), frames_t),
        }


NODE_CLASS_MAPPINGS = {"EnzinoVideo2Prompt": EnzinoVideo2Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"EnzinoVideo2Prompt": "Enzino Video2Prompt (H3)"}
