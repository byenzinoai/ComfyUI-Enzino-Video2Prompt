"""Video reading for Enzino Video2Prompt.

A Clip knows its fps, frame count and shot cuts, and can read any frames
at full resolution. Sources:
  - a video file (PyAV, OpenCV as fallback)
  - a ComfyUI VIDEO object (core Load Video node)
  - an IMAGE batch (e.g. VHS Load Video), assumed 24 fps
Frames are HxWx3 uint8 numpy arrays.
"""

import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")
_TINY = (48, 27)  # size used for cut detection


def even_indices(start, end, count):
    """`count` indices spread evenly over [start, end) (centre of each slice)."""
    total = max(1, int(end) - int(start))
    count = max(1, min(int(count), total))
    step = total / count
    return sorted({int(start) + min(total - 1, int(step * i + step / 2)) for i in range(count)})


def resize_max_side(frame, max_side):
    """Downscale so the longest side is <= max_side, keep aspect, dims multiple of 2."""
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    nw = max(32, int(round(w * scale / 2)) * 2)
    nh = max(32, int(round(h * scale / 2)) * 2)
    if (nw, nh) == (w, h):
        return frame
    try:
        import cv2
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        from PIL import Image
        return np.asarray(Image.fromarray(frame).resize((nw, nh), Image.LANCZOS))


def _tiny_gray(frame):
    try:
        import cv2
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, _TINY, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    except Exception:  # noqa: BLE001
        from PIL import Image
        return np.asarray(Image.fromarray(frame).convert("L").resize(_TINY), dtype=np.float32) / 255.0


def detect_cuts(tiny, fps):
    """Frame indices where a new shot starts (hard cuts, including jump cuts in the same framing,
    e.g. a try-on video where only the outfit changes).

    A cut = one frame-to-frame jump far bigger than the motion just before and after it.
    """
    n = len(tiny)
    if n < 3:
        return []
    diffs = np.array([0.0] + [float(np.mean(np.abs(tiny[i] - tiny[i - 1]))) for i in range(1, n)])
    win = max(4, int(round(fps * 0.4)))
    min_gap = max(1, int(round(fps * 0.4)))
    cuts, last = [], -min_gap
    for i in range(1, n):
        local = np.concatenate([diffs[max(1, i - win):i], diffs[i + 1:min(n, i + win + 1)]])
        if local.size == 0:
            continue
        base = float(np.median(local))
        high = float(np.percentile(local, 90))
        if diffs[i] > 0.018 and diffs[i] > 5.0 * base + 0.006 and diffs[i] > 2.0 * high and i - last >= min_gap:
            cuts.append(i)
            last = i
    return cuts


class Clip:
    fps = 24.0
    total = 0
    cuts = ()

    @property
    def duration(self):
        return self.total / self.fps

    def read(self, indices):
        raise NotImplementedError


class FileClip(Clip):
    """Pass 1 (on creation): count frames and find cuts on tiny copies.
    read(): decode again and keep only the frames asked for."""

    def __init__(self, source):
        self.source = source
        try:
            self._scan_av()
        except Exception as e_av:  # noqa: BLE001
            if not isinstance(source, str):
                raise RuntimeError(f"Could not read the video: {e_av}") from e_av
            try:
                self._scan_cv2()
            except Exception as e_cv:  # noqa: BLE001
                raise RuntimeError(f"Could not read the video. PyAV: {e_av} | OpenCV: {e_cv}") from e_cv
        if self.total == 0:
            raise RuntimeError("No video frame could be decoded.")

    def _open_av(self):
        import av

        if hasattr(self.source, "seek"):
            self.source.seek(0)
        c = av.open(self.source)
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        return c, s

    def _scan_av(self):
        c, s = self._open_av()
        try:
            self.fps = float(s.average_rate or s.guessed_rate or 24) or 24.0
            tiny = [_tiny_gray(f.to_ndarray(format="rgb24")) for f in c.decode(s)]
        finally:
            c.close()
        self.backend = "av"
        self.total = len(tiny)
        self.cuts = detect_cuts(tiny, self.fps)

    def _scan_cv2(self):
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError("cannot open")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 24) or 24.0
        tiny = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            tiny.append(_tiny_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        cap.release()
        self.backend = "cv2"
        self.total = len(tiny)
        self.cuts = detect_cuts(tiny, self.fps)

    def read(self, indices):
        wanted = sorted(set(int(i) for i in indices))
        if not wanted:
            return {}
        out, want, last = {}, set(wanted), wanted[-1]
        if self.backend == "av":
            c, s = self._open_av()
            try:
                for i, f in enumerate(c.decode(s)):
                    if i in want:
                        out[i] = f.to_ndarray(format="rgb24")
                    if i >= last:
                        break
            finally:
                c.close()
        else:
            import cv2

            cap = cv2.VideoCapture(self.source)
            i = 0
            while i <= last:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in want:
                    out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                i += 1
            cap.release()
        return out


class TensorClip(Clip):
    def __init__(self, images, fps=24.0):
        self.images = images
        self.fps = float(fps) or 24.0
        self.total = int(images.shape[0])
        step = max(1, self.total // 2000)  # very long batches: subsample the scan
        tiny = [_tiny_gray(self._frame(i)) for i in range(0, self.total, step)]
        self.cuts = [c * step for c in detect_cuts(tiny, self.fps / step)]

    def _frame(self, i):
        arr = self.images[i].detach().cpu().float().clamp(0, 1).numpy()
        return (arr[..., :3] * 255.0 + 0.5).astype(np.uint8)

    def read(self, indices):
        return {int(i): self._frame(int(i)) for i in indices if 0 <= int(i) < self.total}


def clip_from_file(path):
    return FileClip(path)


def clip_from_images(images, fps=24.0):
    return TensorClip(images, fps)


def clip_from_video_object(video):
    """ComfyUI core VIDEO input (comfy_api VideoInput)."""
    try:
        return FileClip(video.get_stream_source())
    except Exception:  # noqa: BLE001
        comps = video.get_components()
        return TensorClip(comps.images, float(getattr(comps, "frame_rate", 24) or 24))


def fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:04.1f}" if m else f"{s:.1f}s"


def fmt_cut(seconds):
    """H3 cut notation: 00:03.500"""
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    return f"{m:02d}:{seconds - m * 60:06.3f}"
