"""Fit a clip into one H3 video (H3 makes 4–15 s videos).

The Part covers [start, end) frames of the source and knows its shots.
`scale` maps source seconds to target seconds (1.0, or < 1 when a long
clip is squeezed into 15 s).
"""

from dataclasses import dataclass, field

H3_MAX = 15.0
H3_MIN = 4.0
LONG_MODES = ["Fit into 15 s", "Window"]


@dataclass
class Part:
    start: int            # first source frame
    end: int              # source frame after the last one
    fps: float
    cuts: list = field(default_factory=list)   # source frames where a shot starts (inside the part)
    scale: float = 1.0

    @property
    def src_duration(self):
        return (self.end - self.start) / self.fps

    @property
    def duration(self):
        """Length of the video H3 should make."""
        return self.src_duration * self.scale

    def to_target(self, frame):
        return (frame - self.start) / self.fps * self.scale

    def shots(self):
        """[(shot_number, target_start_s, target_end_s)]"""
        bounds = [self.start] + list(self.cuts) + [self.end]
        return [(i + 1, self.to_target(a), self.to_target(b)) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]

    def sample(self, count):
        """`count` frames spread over the part, at least 2 per shot so short shots are seen too."""
        from .video_io import even_indices

        bounds = [self.start] + list(self.cuts) + [self.end]
        total = self.end - self.start
        picked = []
        for a, b in zip(bounds, bounds[1:]):
            n = max(2, round(count * (b - a) / total)) if b - a >= 2 else 1
            picked += even_indices(a, b, n)
        return sorted(set(picked))

    def shot_of(self, frame):
        n = 1
        for c in self.cuts:
            if frame >= c:
                n += 1
        return n


def _cuts_in(cuts, start, end, fps, min_shot=1.0):
    """Cuts strictly inside the range, dropping ones that would leave a shot shorter than min_shot."""
    keep, prev = [], start
    for c in cuts:
        if start < c < end and (c - prev) / fps >= min_shot and (end - c) / fps >= min_shot * 0.5:
            keep.append(c)
            prev = c
    return keep


def plan(total, fps, cuts, mode, win_start=0.0, win_end=0.0):
    """Returns (parts, note) where note explains what was done to a long clip ('' if nothing)."""
    duration = total / fps
    if duration <= H3_MAX + 0.05 and mode != "Window":
        return [Part(0, total, fps, _cuts_in(cuts, 0, total, fps))], ""

    if mode == "Window":
        a = max(0, min(total - 1, int(round(win_start * fps))))
        b = int(round(win_end * fps)) if win_end > win_start else a + int(H3_MAX * fps)
        b = max(a + 1, min(total, b, a + int(H3_MAX * fps)))
        note = f"window {a / fps:.1f}–{b / fps:.1f} s"
        if win_end > win_start and (win_end - win_start) > H3_MAX + 0.05:
            note += " (cut to 15 s)"
        return [Part(a, b, fps, _cuts_in(cuts, a, b, fps))], note

    # default: Fit into 15 s
    scale = H3_MAX / duration
    part = Part(0, total, fps, _cuts_in(cuts, 0, total, fps, min_shot=1.0 / scale), scale)
    return [part], f"{duration:.1f} s squeezed into 15 s"
