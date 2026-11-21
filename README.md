# Enzino Video2Prompt 1.0.0

A video in, a MiniMax H3 prompt out. One ComfyUI node watches a clip with a local Qwen vision model
(Qwen3-VL / Qwen3.5) and writes the prompt H3 needs to recreate it: shot, action in order, camera move,
light, mood and sound, in MiniMax's official H3 prompt format (shots with cut times,
soundscape, music). Always one prompt for one H3 video of 15 s max: longer clips are squeezed into 15 s (cut times scaled) or windowed.

**Full guide:** open `Guide/index.html` from the pack (install, modes, settings, files, troubleshooting).

## Quick install
1. Copy this folder into `ComfyUI/custom_nodes/`.
2. `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Enzino-Video2Prompt\requirements.txt`
3. Restart ComfyUI and load `Workflow/Enzino Video2Prompt.json`.

The model downloads to `ComfyUI/models/VLM/` on first use.

https://enzinoai.com
