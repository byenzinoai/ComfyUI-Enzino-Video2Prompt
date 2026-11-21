"""Instructions given to the vision model, and clean-up of its answer.

The "H3 official" format follows MiniMax's own prompt guide
(github.com/MiniMax-AI/MiniMax-H3, skills/h3-prompt-writing/references/base-en.txt):
  integrated_multimodal_description / overall_soundscape / non_diegetic_music,
  [Shot N] At 00:03.500 cut notation, Motion type + amplitude + speed camera wording,
  and the I2VA first-frame instruction line.
"""

import re

from .video_io import fmt_cut

MODES = ["Faithful", "Motion only"]
LENGTHS = ["Short", "Detailed"]
FORMATS = ["H3 official", "Simple paragraph"]

I2V_LINE = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")

CAMERA_VOCAB = (
    "Camera motion = motion type + optional amplitude + optional speed, written as a natural sentence inside the shot. "
    "Motion types: zooms in/out (lens only), pushes in/pulls out (camera moves), pans left/right, trucks left/right, "
    "tilts up/down, pedestals up/down, arc shot, tracking shot, static shot, shakes slightly/strongly, POV, rolls clockwise/counterclockwise. "
    "Amplitude: 'with small amplitude' / 'with large amplitude'. Speed: 'at slow speed' / 'at fast speed'. "
    "Omit amplitude and speed when medium/normal. Only describe the camera motion you actually see; a still camera is a static shot."
)

SYSTEM_H3 = (
    "You turn a reference video clip into a MiniMax H3 prompt that recreates it. You only describe what is visible "
    "in the frames; you can not hear the clip.\n"
    "Output EXACTLY three fields, in this order, separated by one blank line, with these exact names:\n"
    "integrated_multimodal_description: [Shot 1] ...\n\n"
    "overall_soundscape: ...\n\n"
    "non_diegetic_music: ...\n\n"
    "Rules for integrated_multimodal_description:\n"
    "- Start [Shot 1] with the overall style and the initial composition, e.g. '[Shot 1] Live-action, cinematic, a medium-wide shot frames...'. "
    "Styles: Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film.\n"
    "- Describe along the timeline: subject appearance and position, setting and key props, actions and reactions in order, camera motion.\n"
    "- The first shot has no timestamp. Every later shot starts with its cut time, exactly as given to you: "
    "'[Shot 2] At 00:03.500, the camera cuts to ...'. Use exactly the shots and cut times you are given, no more, no less.\n"
    f"- {CAMERA_VOCAB}\n"
    "- Visible on-screen text goes in double quotes, verbatim.\n"
    "- Never invent dialogue (you cannot hear it). If someone visibly talks, say they speak without quoting words.\n"
    "- Concrete visual words only; no plot summary, no abstract words like 'beautiful'.\n"
    "Rules for overall_soundscape: 1 to 4 English sentences, one paragraph: ambience, physical action sounds, non-verbal human sounds "
    "(footsteps, fabric, wind, breathing...), inferred from what is seen.\n"
    "Rules for non_diegetic_music: background music only the audience hears, 1 to 3 sentences about instrumentation, tempo, rhythm and "
    "dynamics (no mood words). Write N/A unless the clip is clearly an edited montage, ad, fashion or music-video style piece.\n"
    "No markdown, no bold, no headings, no preamble, nothing after non_diegetic_music."
)

SYSTEM_SIMPLE = (
    "You are a film director's assistant who writes prompts for MiniMax H3, a text-to-video and image-to-video model. "
    "You watch a reference clip and write ONE prompt that lets H3 recreate it. H3 understands plain, concrete, "
    "present-tense English best: what is seen, what happens in order, how the camera moves, the light, the mood. "
    f"{CAMERA_VOCAB} No poetry, no lists, no headings, no quotes, no markdown, no preamble."
)

LENGTH_RULES = {
    "Short": "Keep the main description to about 60 to 100 words.",
    "Detailed": "Make the main description about 140 to 220 words.",
}


def shot_schedule(part):
    shots = part.shots()
    lines = [f"The target video lasts {part.duration:.2f} seconds and has {len(shots)} shot(s):"]
    for n, a, b in shots:
        if n == 1:
            lines.append(f"- [Shot 1] from the start to {b:.2f} s (no timestamp)")
        else:
            lines.append(f"- [Shot {n}] At {fmt_cut(a)} (until {b:.2f} s)")
    return "\n".join(lines)


def build_user_prompt(fmt, mode, length, part, frame_lines, sound, notes, squeeze_note=""):
    h3 = fmt == "H3 official"
    parts = [f"These frames come from the reference clip, in order:\n{frame_lines}", shot_schedule(part)]
    if squeeze_note:
        parts.append(squeeze_note)

    if mode == "Motion only":
        parts.append(
            "The user animates their OWN first image (image-to-video): the picture already exists. "
            + ("Begin [Shot 1] with 'the subject shown in <Picture 1>' and say its look, framing and setting are preserved, "
               "then describe only what happens next. " if h3 else "")
            + "Do NOT describe faces, hair, clothing, colours or the setting in detail: name each subject with a plain generic "
            "word for what it really is (person, woman, man, dog, car...). Describe every action in order with precise verbs, body "
            "parts and directions, the camera motion, the pace, and secondary motion (hair, fabric, smoke, water, light)."
        )
    else:
        parts.append(
            "Describe the clip so H3 can rebuild it from text alone: style, framing, subjects and their look, setting, "
            "actions in order, camera motion, light and colour."
        )
    parts.append(LENGTH_RULES.get(length, LENGTH_RULES["Short"]))

    if h3:
        skel = []
        for n, a, _b in part.shots():
            skel.append("[Shot 1] ..." if n == 1 else f"[Shot {n}] At {fmt_cut(a)}, the camera cuts to ...")
        parts.append(
            "Fill in this skeleton. Replace each ... with your description and keep every marker exactly as written "
            "(one [Shot N] per shot, in this order, describing the frames labelled with that shot):\n"
            f"integrated_multimodal_description: {' '.join(skel)}\n\noverall_soundscape: ...\n\nnon_diegetic_music: ..."
        )
        if not sound:
            parts.append("The video must be silent: write 'overall_soundscape: N/A' and 'non_diegetic_music: N/A'.")
    else:
        if sound:
            parts.append("End with one short sentence about the sound the scene would have, inferred from what is seen. No dialogue.")
        if len(part.shots()) > 1:
            parts.append("Mention each cut where it happens ('then the shot cuts to ...').")
        parts.append("Write one flowing paragraph.")

    notes = (notes or "").strip()
    if notes:
        parts.append("Apply these changes from the user (they win over what is seen in the clip): " + notes)
    parts.append("Answer with the prompt only.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- clean-up

def _strip_common(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).replace("</think>", "")
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#+\s.*$", "", text, flags=re.M)
    return text.strip()


_PREFIX = re.compile(r"^\s*(?:prompt|h3 prompt|video prompt|here is (?:the|your) prompt)\s*[:：]\s*", re.I)


def clean_simple(text):
    text = _PREFIX.sub("", _strip_common(text))
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", l).strip() for l in text.splitlines()]
    text = re.sub(r"\s{2,}", " ", " ".join(l for l in lines if l)).strip()
    if len(text) > 1 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text


def _one_paragraph(s):
    lines = [re.sub(r"^\s*(?:[-*•])\s+", "", l).strip() for l in s.splitlines()]
    return re.sub(r"\s{2,}", " ", " ".join(l for l in lines if l)).strip()


def _end_clean(t):
    """Drop a sentence cut off by the token limit."""
    t = t.strip()
    if t and t[-1] not in ".!?\"'”)" and "." in t:
        t = t[: t.rfind(".") + 1]
    return t


def _tidy_side(t):
    """Sound/music fields: no shot markers (small models sometimes repeat the timeline there)."""
    t = (t or "").split("[Shot")[0].strip()
    t = _end_clean(t)
    return t or "N/A"


def clean_h3(text, i2v=False, sound=True):
    """Rebuild the three official fields whatever small formatting slips the model made."""
    text = _strip_common(text)
    text = text.replace(I2V_LINE, "")
    # Field names anywhere (small models sometimes put them mid-line or write "Non-diegetic music:")
    pattern = re.compile(r"(?i)[`'\"]?\b(integrated[ _-]multimodal[ _-]description|overall[ _-]soundscape|"
                         r"non[ _-]diegetic[ _-]music)\b[`'\"]?\s*[:：]\s*")
    found = {}
    marks = list(pattern.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        key = re.sub(r"[ -]", "_", m.group(1).lower())
        if key not in found:
            found[key] = _one_paragraph(text[m.end():end])
    if not marks:  # model ignored the format: everything is the description
        found[FIELDS[0]] = _one_paragraph(text)

    desc = found.get(FIELDS[0], "")
    if desc and not desc.startswith("[Shot 1]"):
        desc = "[Shot 1] " + re.sub(r"^\[?shot\s*1\]?\s*[:.-]?\s*", "", desc, flags=re.I)
    # "[Shot 2] At 3.5s" -> "[Shot 2] At 00:03.500"
    desc = re.sub(r"\[Shot (\d+)\]\s*At\s+(\d+(?:\.\d+)?)\s*s(?:econds)?\b",
                  lambda m: f"[Shot {m.group(1)}] At {fmt_cut(float(m.group(2)))}", desc)
    sound_txt = _tidy_side(found.get(FIELDS[1]))
    music_txt = _tidy_side(found.get(FIELDS[2]))
    desc = _end_clean(desc)
    if not sound:
        sound_txt = music_txt = "N/A"
    if music_txt.lower().rstrip(".") in ("n/a", "na", "none", "no music"):
        music_txt = "N/A"

    out = f"{FIELDS[0]}: {desc}\n\n{FIELDS[1]}: {sound_txt}\n\n{FIELDS[2]}: {music_txt}"
    if i2v:
        out = I2V_LINE + "\n\n" + out
    return out


def word_count(prompt):
    m = re.search(FIELDS[0] + r":\s*(.*?)(?:\n\n" + FIELDS[1] + r"|$)", prompt, flags=re.S)
    body = m.group(1) if m else prompt
    return len(body.split())
