import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
from tqdm import tqdm
from PIL import Image
from scenedetect import AdaptiveDetector, SceneManager, open_video
from scenedetect.backends.pyav import VideoStreamAv

try:
    from decord import VideoReader, cpu, gpu
    from decord._ffi.base import DECORDError
    _DECORD_AVAILABLE = True
except Exception:
    VideoReader = Any
    DECORDError = Exception
    _DECORD_AVAILABLE = False

    def cpu(_index=0):
        return None

    def gpu(_index=0):
        return None


class _NumpyFrame:
    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    @property
    def shape(self):
        return self._array.shape


class _NumpyBatch:
    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array


class PyAVVideoReader:
    """Small decord-compatible reader used when decord is unavailable.

    This fallback favors portability over speed. It supports only the methods
    CutClaw uses: len(), get_avg_fps(), __getitem__(), and get_batch().
    """

    def __init__(
        self,
        video_path: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **_: Any,
    ):
        self.video_path = video_path
        self.height = height
        self.width = width

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            self._fps = float(rate) if rate else 24.0
            self._native_width = int(stream.codec_context.width or stream.width or 0)
            self._native_height = int(stream.codec_context.height or stream.height or 0)
            if stream.frames:
                self._num_frames = int(stream.frames)
            elif container.duration:
                self._num_frames = max(0, int((container.duration / av.time_base) * self._fps))
            else:
                self._num_frames = 0

    def __len__(self) -> int:
        return self._num_frames

    def get_avg_fps(self) -> float:
        return self._fps

    def _resize_if_needed(self, array: np.ndarray) -> np.ndarray:
        if self.height is None or self.width is None:
            return array
        image = Image.fromarray(array)
        image = image.resize((int(self.width), int(self.height)), Image.Resampling.BILINEAR)
        return np.asarray(image)

    def get_batch(self, indices: List[int]) -> _NumpyBatch:
        if not indices:
            return _NumpyBatch(np.empty((0, 0, 0, 3), dtype=np.uint8))

        wanted = {int(i) for i in indices if int(i) >= 0}
        max_index = max(wanted)
        frames: Dict[int, np.ndarray] = {}

        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in wanted:
                    arr = frame.to_ndarray(format="rgb24")
                    frames[frame_idx] = self._resize_if_needed(arr)
                    if len(frames) == len(wanted):
                        break
                if frame_idx > max_index:
                    break

        if not frames:
            raise IndexError(f"No requested frames were decoded from {self.video_path}")

        fallback = frames[sorted(frames.keys())[-1]]
        ordered = [frames.get(int(i), fallback) for i in indices]
        return _NumpyBatch(np.stack(ordered, axis=0))

    def __getitem__(self, index: int) -> _NumpyFrame:
        return _NumpyFrame(self.get_batch([int(index)]).asnumpy()[0])


def _save_sampled_frames_to_disk_pyav_streaming(
    video_reader: PyAVVideoReader,
    frame_indices: List[int],
    frames_dir: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
) -> List[str]:
    """Save sampled frames using one forward decode pass on the source video."""
    file_ext = image_format.lower()
    saved_paths: List[str] = []
    quality = max(1, min(100, int(jpeg_quality)))

    targets = [int(i) for i in frame_indices if int(i) >= 0]
    if not targets:
        return saved_paths

    progress = tqdm(total=len(targets), desc="Saving frames", unit="frame")
    try:
        with av.open(video_reader.video_path) as container:
            stream = container.streams.video[0]
            target_pos = 0
            current_target = targets[target_pos]

            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < current_target:
                    continue

                if frame_idx > current_target:
                    while target_pos < len(targets) and targets[target_pos] < frame_idx:
                        target_pos += 1
                        progress.update(1)
                    if target_pos >= len(targets):
                        break
                    current_target = targets[target_pos]
                    if frame_idx < current_target:
                        continue

                arr = frame.to_ndarray(format="rgb24")
                arr = video_reader._resize_if_needed(arr)

                while target_pos < len(targets) and targets[target_pos] == frame_idx:
                    out_path = os.path.join(frames_dir, f"frame_{target_pos:06d}.{file_ext}")
                    image = Image.fromarray(arr)
                    if file_ext in {"jpg", "jpeg"}:
                        image.save(out_path, quality=quality)
                    else:
                        image.save(out_path)
                    saved_paths.append(out_path)
                    target_pos += 1
                    progress.update(1)
                    if target_pos >= len(targets):
                        break

                if target_pos >= len(targets):
                    break
                current_target = targets[target_pos]
    finally:
        progress.close()

    if len(saved_paths) != len(targets):
        raise RuntimeError(
            f"Expected to cache {len(targets)} sampled frames, "
            f"but only saved {len(saved_paths)} from {video_reader.video_path}"
        )

    return saved_paths


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


_decord_ctx = None


def _get_decord_ctx():
    """Return GPU context if decord was compiled with CUDA support, otherwise CPU.

    The check is cached after the first call via _create_decord_reader, which
    has the video path needed to trigger decord's CUDA validation.
    """
    if not _DECORD_AVAILABLE:
        return None
    global _decord_ctx
    if _decord_ctx is None:
        return gpu(0)  # tentative; _create_decord_reader will confirm
    return _decord_ctx


def _adjust_scene_boundaries(scenes: List[List[int]]) -> List[List[int]]:
    if not scenes or len(scenes) <= 1:
        return scenes

    adjusted = [scenes[0]]
    for i in range(1, len(scenes)):
        start_frame = adjusted[-1][1] + 1
        end_frame = scenes[i][1]
        if start_frame >= end_frame:
            # zero-duration shot: merge into previous
            adjusted[-1][1] = end_frame
        else:
            adjusted.append([start_frame, end_frame])
    return adjusted


def _timecode_to_seconds(timecode: str) -> float:
    hours, minutes, seconds_milliseconds = timecode.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def _create_decord_reader(
    video_path: str,
    target_resolution: Optional[Tuple[int, int]] = None,
) -> VideoReader:
    global _decord_ctx

    if not _DECORD_AVAILABLE:
        if target_resolution is None:
            return PyAVVideoReader(video_path)
        if isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
            return PyAVVideoReader(
                video_path,
                height=int(target_resolution[0]),
                width=int(target_resolution[1]),
            )

        short_side = int(target_resolution[0]) if isinstance(target_resolution, (tuple, list)) else int(target_resolution)
        probe = PyAVVideoReader(video_path)
        native_h, native_w = probe[0].shape[:2]
        if native_h <= native_w:
            target_h = short_side
            target_w = int(round(native_w * short_side / native_h / 2) * 2)
        else:
            target_w = short_side
            target_h = int(round(native_h * short_side / native_w / 2) * 2)
        return PyAVVideoReader(video_path, height=target_h, width=target_w)

    def _make_reader(path, ctx, **kwargs):
        """Try to open VideoReader; fall back to CPU if decord lacks CUDA support."""
        global _decord_ctx
        try:
            vr = VideoReader(path, ctx=ctx, **kwargs)
            _decord_ctx = ctx  # confirmed working
            return vr
        except DECORDError as e:
            if 'CUDA not enabled' in str(e):
                _decord_ctx = cpu(0)
                return VideoReader(path, ctx=_decord_ctx, **kwargs)
            raise

    ctx = _get_decord_ctx()
    if target_resolution is None:
        return _make_reader(video_path, ctx, num_threads=8)

    # Parse target_resolution as a short-side constraint (int) or explicit (H, W)
    if isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
        # Explicit (H, W) — use as-is
        return _make_reader(video_path, ctx,
                            height=int(target_resolution[0]),
                            width=int(target_resolution[1]),
                            num_threads=16)

    # Short-side mode: read native resolution first, then scale proportionally
    short_side = int(target_resolution[0]) if isinstance(target_resolution, (tuple, list)) else int(target_resolution)
    probe = _make_reader(video_path, ctx)
    native_h, native_w = probe[0].shape[:2]
    del probe
    ctx = _decord_ctx or cpu(0)  # use confirmed context after probe

    if native_h <= native_w:
        # Height is the short side
        target_h = short_side
        target_w = int(round(native_w * short_side / native_h / 2) * 2)  # keep even
    else:
        # Width is the short side
        target_w = short_side
        target_h = int(round(native_h * short_side / native_w / 2) * 2)  # keep even

    return _make_reader(video_path, ctx, height=target_h, width=target_w)


def _sampled_frame_cache_dir(
    frames_dir: str,
    target_fps: float,
    short_side: int,
    image_format: str,
) -> str:
    fps_str = str(int(target_fps)) if float(target_fps).is_integer() else str(target_fps).replace(".", "p")
    return os.path.join(frames_dir, f"sampled_frames_{fps_str}fps_{int(short_side)}p_{image_format.lower()}")


def _list_cached_frame_paths(
    frames_dir: str,
    image_format: str,
    expected_count: Optional[int] = None,
) -> List[str]:
    if not os.path.isdir(frames_dir):
        return []

    suffixes = {f".{image_format.lower()}"}
    if image_format.lower() == "jpg":
        suffixes.add(".jpeg")

    paths = sorted(
        os.path.join(frames_dir, name)
        for name in os.listdir(frames_dir)
        if name.startswith("frame_") and os.path.splitext(name)[1].lower() in suffixes
    )
    if expected_count is not None and len(paths) != int(expected_count):
        return []
    return paths


def get_cached_sampled_frame_paths(
    frames_dir: str,
    target_fps: float,
    short_side: int,
    image_format: str,
    expected_count: Optional[int] = None,
) -> tuple[Optional[str], List[str]]:
    """Resolve sampled-frame cache directory + ordered frame paths for a known cache spec."""
    frame_cache_dir = _sampled_frame_cache_dir(
        frames_dir=frames_dir,
        target_fps=target_fps,
        short_side=short_side,
        image_format=image_format,
    )
    frame_paths = _list_cached_frame_paths(
        frame_cache_dir,
        image_format=image_format,
        expected_count=expected_count,
    )
    return frame_cache_dir, frame_paths


def _resize_array_short_side(array: np.ndarray, target_short_side: Optional[int]) -> np.ndarray:
    if not target_short_side or target_short_side <= 0:
        return array

    height, width = array.shape[:2]
    current_short = min(height, width)
    if current_short <= target_short_side:
        return array

    if height <= width:
        target_h = int(target_short_side)
        target_w = int(round(width * target_short_side / height / 2) * 2)
    else:
        target_w = int(target_short_side)
        target_h = int(round(height * target_short_side / width / 2) * 2)

    image = Image.fromarray(array)
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(image)


def load_cached_sampled_frames(
    frame_paths: List[str],
    sampled_indices: List[int],
    target_short_side: Optional[int] = None,
) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for sampled_idx in sampled_indices:
        if sampled_idx < 0 or sampled_idx >= len(frame_paths):
            continue
        frame_path = frame_paths[sampled_idx]
        if not os.path.exists(frame_path):
            continue
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            arr = np.asarray(rgb)
        arrays.append(_resize_array_short_side(arr, target_short_side))
    return arrays


def _save_sampled_frames_to_disk(
    video_reader: VideoReader,
    frame_indices: List[int],
    frames_dir: str,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    batch_size: int = 128,
) -> List[str]:
    file_ext = image_format.lower()
    if file_ext not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    _ensure_dir(frames_dir)

    for filename in os.listdir(frames_dir):
        if filename.startswith("frame_") and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(frames_dir, filename))

    if not frame_indices:
        return []

    if isinstance(video_reader, PyAVVideoReader):
        return _save_sampled_frames_to_disk_pyav_streaming(
            video_reader=video_reader,
            frame_indices=frame_indices,
            frames_dir=frames_dir,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )

    saved_paths: List[str] = []
    quality = max(1, min(100, int(jpeg_quality)))
    batch_size = max(1, int(batch_size))

    progress = tqdm(total=len(frame_indices), desc="Saving frames", unit="frame")
    try:
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(len(frame_indices), batch_start + batch_size)
            sampled = video_reader.get_batch(frame_indices[batch_start:batch_end]).asnumpy()
            for local_idx, frame in enumerate(sampled):
                out_idx = batch_start + local_idx
                out_path = os.path.join(frames_dir, f"frame_{out_idx:06d}.{file_ext}")
                image = Image.fromarray(frame)
                if file_ext in {"jpg", "jpeg"}:
                    image.save(out_path, quality=quality)
                else:
                    image.save(out_path)
                saved_paths.append(out_path)
                progress.update(1)
    finally:
        progress.close()

    return saved_paths


def _run_scenedetect(
    video_path: str,
    threshold: float,
    min_scene_len: int,
    end_frame: Optional[int],
    frame_skip: int = 0,
    start_frame: int = 0,
    warmup_start_frame: int = 0,
) -> List:
    """Run scenedetect on [warmup_start_frame, end_frame), but only return scenes >= start_frame."""
    from scenedetect.frame_timecode import FrameTimecode

    video = VideoStreamAv(video_path)
    manager = SceneManager()
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len,
        )
    )

    if warmup_start_frame > 0:
        video.seek(warmup_start_frame)

    detect_kwargs = {}
    if frame_skip > 0:
        detect_kwargs["frame_skip"] = frame_skip
    if end_frame is not None:
        detect_kwargs["end_time"] = FrameTimecode(end_frame, fps=video.frame_rate)

    manager.detect_scenes(video, **detect_kwargs)

    scene_list = manager.get_scene_list()

    # Filter out scenes that belong to the warmup region (before start_frame)
    if warmup_start_frame < start_frame:
        scene_list = [s for s in scene_list if s[0].get_frames() >= start_frame]

    return scene_list


def _run_scenedetect_segment(args: tuple) -> List:
    """Worker function for parallel scenedetect. Returns scene list for one segment."""
    video_path, threshold, min_scene_len, start_frame, end_frame, frame_skip, warmup_frames = args
    warmup_start_frame = max(0, start_frame - warmup_frames)
    return _run_scenedetect(
        video_path, threshold, min_scene_len, end_frame, frame_skip,
        start_frame=start_frame, warmup_start_frame=warmup_start_frame,
    )


def _run_scenedetect_parallel(
    video_path: str,
    threshold: float,
    min_scene_len: int,
    total_frames: int,
    frame_skip: int = 0,
    num_workers: int = 8,
) -> List:
    """Split video into segments and run scenedetect in parallel across processes.

    Each segment starts decoding `warmup_frames` before the logical segment boundary
    so that AdaptiveDetector's sliding window is warmed up before the region of interest.
    Scenes detected in the warmup region are discarded.
    """
    # warmup must cover: AdaptiveDetector window (window_width=2 → 5 frames) + min_scene_len,
    # all scaled by (frame_skip+1) to account for skipped frames.
    warmup_frames = (min_scene_len + 5) * (frame_skip + 1)

    segment_size = total_frames // num_workers
    segments = []
    for i in range(num_workers):
        seg_start = i * segment_size
        seg_end = total_frames if i == num_workers - 1 else (i + 1) * segment_size
        segments.append((video_path, threshold, min_scene_len, seg_start, seg_end, frame_skip, warmup_frames))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_run_scenedetect_segment, segments))

    all_scenes = []
    for scene_list in results:
        all_scenes.extend(scene_list)

    all_scenes.sort(key=lambda s: s[0].get_frames())
    return all_scenes


def scenedetect_extract_and_detect(
    video_path: str,
    frames_dir: str,
    target_fps: float = 2.0,
    target_resolution: Optional[int] = None,
    threshold: float = 3.0,
    min_scene_len: int = 15,
    save_frames_to_disk: bool = False,
    image_format: str = "jpg",
    jpeg_quality: int = 95,
    max_minutes: Optional[float] = None,
    num_workers: int = 1,
    cache_sampled_frames: bool = False,
    cache_resolution: Optional[int] = None,
    cache_image_format: str = "jpg",
    cache_jpeg_quality: int = 80,
    cache_batch_size: int = 128,
) -> dict:
    _ensure_dir(frames_dir)

    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    meta_video = open_video(video_path)
    video_fps = float(meta_video.frame_rate)
    total_frames = int(meta_video.duration.get_frames())

    if max_minutes is not None:
        end_frame = min(int(max_minutes * 60 * video_fps), total_frames)
    else:
        end_frame = total_frames

    # 与旧版等间隔采样逻辑保持一致，返回采样索引而不落盘抽帧
    frame_indices: List[int] = []
    sample_interval = video_fps / target_fps
    current_frame = 0.0
    while int(current_frame) < end_frame:
        frame_indices.append(int(current_frame))
        current_frame += sample_interval

    # Compute frame_skip so scenedetect processes at roughly target_fps
    frame_skip = max(0, round(video_fps / target_fps) - 1)

    shot_scenes_path = os.path.join(frames_dir, "shot_scenes.txt")
    video_reader = _create_decord_reader(video_path, target_resolution)

    if os.path.exists(shot_scenes_path):
        print(f"[SceneDetect] Found existing shot_scenes.txt, skipping detection")
        scene_list = None  # will load from file below
    else:
        print(f"[SceneDetect] Running PySceneDetect (frame_skip={frame_skip}, num_workers={num_workers})")
        if num_workers > 1:
            scene_list = _run_scenedetect_parallel(
                video_path,
                threshold,
                min_scene_len,
                end_frame,
                frame_skip=frame_skip,
                num_workers=num_workers,
            )
        else:
            scene_list = _run_scenedetect(
                video_path,
                threshold,
                min_scene_len,
                end_frame if max_minutes is not None else None,
                frame_skip=frame_skip,
            )
    sample_fps = float(target_fps)

    if len(video_reader) > 0:
        first_frame = video_reader[0].asnumpy()
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    else:
        height, width = 0, 0

    if len(video_reader) > 0:
        frame_indices = [idx for idx in frame_indices if idx < len(video_reader)]

    frame_paths: List[str] = []
    frame_cache_dir: Optional[str] = None
    if cache_sampled_frames:
        cache_short_side = int(cache_resolution or target_resolution or 720)
        frame_cache_dir = _sampled_frame_cache_dir(
            frames_dir=frames_dir,
            target_fps=target_fps,
            short_side=cache_short_side,
            image_format=cache_image_format,
        )
        frame_paths = _list_cached_frame_paths(
            frame_cache_dir,
            image_format=cache_image_format,
            expected_count=len(frame_indices),
        )
        if frame_paths:
            print(f"[SceneDetect] Reusing sampled frame cache: {frame_cache_dir} ({len(frame_paths)} frames)")
        else:
            print(
                f"[SceneDetect] Building sampled frame cache in {frame_cache_dir} "
                f"({len(frame_indices)} frames @ {target_fps}fps, <= {cache_short_side}p)"
            )
            cache_reader = _create_decord_reader(video_path, cache_short_side)
            frame_paths = _save_sampled_frames_to_disk(
                video_reader=cache_reader,
                frame_indices=frame_indices,
                frames_dir=frame_cache_dir,
                image_format=cache_image_format,
                jpeg_quality=cache_jpeg_quality,
                batch_size=cache_batch_size,
            )
            del cache_reader
    elif save_frames_to_disk:
        print(f"[SceneDetect] Saving sampled frames to disk in {frames_dir}")
        frame_paths = _save_sampled_frames_to_disk(
            video_reader=video_reader,
            frame_indices=frame_indices,
            frames_dir=frames_dir,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            batch_size=cache_batch_size,
        )

    scenes: List[List[int]] = []
    if scene_list is not None:
        for scene in scene_list:
            start_sec = _timecode_to_seconds(scene[0].get_timecode())
            end_sec_scene = _timecode_to_seconds(scene[1].get_timecode())
            start_frame = int(start_sec * sample_fps)
            end_frame_i = int(end_sec_scene * sample_fps)
            scenes.append([start_frame, end_frame_i])

        scenes = _adjust_scene_boundaries(scenes)

        if scenes:
            np.savetxt(shot_scenes_path, np.array(scenes), fmt="%d")
        else:
            np.savetxt(shot_scenes_path, np.array([]).reshape(0, 2), fmt="%d")
    else:
        # Load existing scenes from file
        raw = np.loadtxt(shot_scenes_path, dtype=int)
        if raw.ndim == 1 and len(raw) == 2:
            scenes = [raw.tolist()]
        elif raw.ndim == 2:
            scenes = raw.tolist()
        else:
            scenes = []

    print(f"[SceneDetect] Completed: {len(frame_indices)} sampled indices, {len(scenes)} scenes")

    return {
        "num_frames": len(frame_indices),
        "sample_fps": float(sample_fps),
        "height": int(height),
        "width": int(width),
        "video_reader": video_reader,
        "frame_indices": frame_indices,
        "frame_paths": frame_paths,
        "frame_cache_dir": frame_cache_dir,
        "save_frames_to_disk": bool(save_frames_to_disk),
        "shot_scenes_path": shot_scenes_path,
        "scenes": scenes,
        "shot_detection_fps": float(sample_fps),
        "shot_detection_model": "scenedetect",
    }


def decode_video_to_frames(
    video_path: str,
    frames_dir: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    max_minutes: Optional[float] = None,
    shot_detection_threshold: float = 3.0,
    shot_detection_min_scene_len: int = 15,
    save_frames_to_disk: bool = False,
    image_format: str = "jpg",
    jpeg_quality: int = 80,
    num_workers: int = 16,
    cache_sampled_frames: bool = False,
    cache_resolution: Optional[int] = None,
    cache_image_format: str = "jpg",
    cache_jpeg_quality: int = 80,
    cache_batch_size: int = 128,
) -> Dict[str, Any]:
    fps = float(target_fps) if target_fps is not None else 2.0
    if fps <= 0:
        fps = 2.0

    resolution = target_resolution
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        resolution = (int(resolution[0]), int(resolution[1]))
    elif isinstance(resolution, (tuple, list)) and len(resolution) == 1:
        resolution = int(resolution[0])

    return scenedetect_extract_and_detect(
        video_path=video_path,
        frames_dir=frames_dir,
        target_fps=fps,
        target_resolution=resolution,
        threshold=float(shot_detection_threshold),
        min_scene_len=int(shot_detection_min_scene_len),
        save_frames_to_disk=bool(save_frames_to_disk),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        max_minutes=max_minutes,
        num_workers=num_workers,
        cache_sampled_frames=cache_sampled_frames,
        cache_resolution=cache_resolution,
        cache_image_format=cache_image_format,
        cache_jpeg_quality=cache_jpeg_quality,
        cache_batch_size=cache_batch_size,
    )
