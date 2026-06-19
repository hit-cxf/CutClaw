"""
Shared media utilities for the VideoCuttingAgent pipeline.
Covers: JSON parsing, SRT parsing, time conversion, image encoding, shot scene I/O.
"""

import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def parse_json_safely(text: Optional[str]) -> Optional[Dict]:
    """Robustly parse a JSON string, stripping Markdown code fences if present."""
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to 'HH:MM:SS.s' string (one decimal place)."""
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def hhmmss_to_seconds(time_str: str, fps: float = 24.0) -> float:
    """Convert time string to seconds.

    Supported formats:
    - HH:MM:SS or HH:MM:SS.mmm  (standard)
    - HH:MM:SS:FF               (with frame number, uses fps parameter)
    - MM:SS or MM:SS.mmm
    - plain seconds as string
    """
    if not time_str:
        return 0.0
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    try:
        if len(parts) == 4:
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return h * 3600 + m * 60 + s + (f / fps)
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt_file(srt_path: str) -> List[Dict]:
    """
    Parse an SRT file into a list of subtitle dicts.

    Each dict has keys: start_sec, end_sec, speaker (or None), text.
    """
    if not os.path.exists(srt_path):
        return []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    subtitles = []
    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            time_match = re.match(
                r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})',
                lines[1]
            )
            if not time_match:
                continue
            text = ' '.join(lines[2:]).strip()
            speaker = None
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2).strip()
            subtitles.append({
                'start_sec': hhmmss_to_seconds(time_match.group(1)),
                'end_sec': hhmmss_to_seconds(time_match.group(2)),
                'speaker': speaker,
                'text': text,
            })
        except Exception:
            continue
    return subtitles


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an SRT file and return a mapping '{startSec}_{endSec}' -> 'subtitle text'.
    Timestamps are truncated to integer seconds.
    """
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}
    with open(srt_path, 'r', encoding='utf-8') as fh:
        lines = [l.rstrip('\n') for l in fh]

    idx = 0
    n = len(lines)
    while idx < n:
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break
        if '-->' not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split('-->')]
        start_sec = int(hhmmss_to_seconds(start_ts))
        end_sec = int(hhmmss_to_seconds(end_ts))
        idx += 1
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = ' '.join(subtitle_lines)
        key = f'{start_sec}_{end_sec}'
        result[key] = result[key] + ' ' + subtitle if key in result else subtitle
        idx += 1
    return result


def get_subtitles_in_range(subtitles: List[Dict], start: float, end: float) -> List[Dict]:
    """Return subtitle entries that overlap [start, end]."""
    return [s for s in subtitles if s['end_sec'] >= start and s['start_sec'] <= end]


def format_subtitles(subtitles: List[Dict]) -> str:
    """Format subtitle list as dialogue lines."""
    if not subtitles:
        return 'No dialogue.'
    lines = [
        f"[{s.get('speaker', 'Unknown')}]: \"{s['text']}\""
        for s in subtitles if s.get('text')
    ]
    return '\n'.join(lines) if lines else 'No dialogue.'


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Encode a PIL Image to a base64 JPEG string."""
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def array_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a numpy uint8 RGB array (H, W, C) to a base64 JPEG string."""
    return pil_to_base64(Image.fromarray(frame), quality=quality)


# ---------------------------------------------------------------------------
# Shot scene file I/O
# ---------------------------------------------------------------------------

def parse_shot_scenes(shot_scenes_path: str) -> List[Tuple[int, int]]:
    """Parse a shot_scenes.txt file into a list of (start_frame, end_frame) tuples."""
    scenes: List[Tuple[int, int]] = []
    if not os.path.isfile(shot_scenes_path):
        return scenes
    with open(shot_scenes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scenes.append((int(parts[0]), int(parts[1])))
    return scenes


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def natural_sort_key(s: str) -> List:
    """Key function for natural (human) sort order (e.g. clip_2 before clip_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


# ---------------------------------------------------------------------------
# Screenwriter helpers
# ---------------------------------------------------------------------------

def load_scene_summaries(scene_folder_path: str) -> tuple[str, int]:
    """Load scene_caption.scene_summary from all scene JSON files in a folder.

    Skips non-usable scenes and scenes with importance_score < 3.

    Returns:
        (concatenated scene summaries, number of loaded scenes)
    """
    scene_summaries = []

    scene_files = [f for f in os.listdir(scene_folder_path)
                   if f.startswith('scene_') and f.endswith('.json')]

    def _scene_number(filename: str) -> int:
        try:
            return int(filename.replace('scene_', '').replace('.json', ''))
        except ValueError:
            return float('inf')

    scene_files.sort(key=_scene_number)

    for filename in scene_files:
        filepath = os.path.join(scene_folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)

            video_analysis = scene_data.get('video_analysis', {})
            scene_caption = video_analysis.get('scene_caption', {})
            scene_classification = scene_caption.get('scene_classification', {})

            if not scene_classification.get('is_usable', True):
                print(f"Skipping {filename}: not usable ({scene_classification.get('unusable_reason', 'unknown')})")
                continue

            importance_score = scene_classification.get('importance_score', 5)
            if importance_score < 3:
                print(f"Skipping {filename}: importance_score ({importance_score}) below threshold (3)")
                continue

            scene_summary = scene_caption.get('scene_summary', {})
            if not scene_summary:
                continue

            scene_id = scene_data.get('scene_id', 'Unknown')
            time_range = scene_data.get('time_range', {})
            start_time = time_range.get('start_seconds', 'N/A')
            end_time = time_range.get('end_seconds', 'N/A')

            narrative = scene_summary.get('narrative', '')
            key_event = scene_summary.get('key_event', '')
            location = scene_summary.get('location', '')
            time_state = scene_summary.get('time', '')

            summary_text = (
                f"[Scene {scene_id}] ({start_time} - {end_time})\n"
                f"Location: {location}, Time: {time_state}\n"
                f"Key Event: {key_event}\n"
                f"Narrative: {narrative}\n"
            )
            scene_summaries.append(summary_text)

        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")
            continue

    total_scene_files = len(scene_files)
    print(f"Loaded {len(scene_summaries)} scene summaries (out of {total_scene_files} files) from {scene_folder_path}")
    return "\n".join(scene_summaries), total_scene_files


def parse_structure_proposal_output(output: str) -> Optional[Dict]:
    """Parse structure proposal JSON from LLM output.

    Expected format::

        {
            "overall_theme": "...",
            "narrative_logic": "...",
            "emotion": "...",
            "related_scenes": [list of int scene indices]
        }

    Returns parsed dict or None on failure.
    """
    def _validate(data) -> bool:
        if not isinstance(data, dict):
            return False
        for field in ('overall_theme', 'narrative_logic', 'emotion', 'related_scenes'):
            if field not in data:
                print(f"Warning: Missing required field '{field}'")
                return False
        if not isinstance(data['related_scenes'], list):
            print(f"Warning: 'related_scenes' must be a list")
            return False
        for idx, scene_id in enumerate(data['related_scenes']):
            if not isinstance(scene_id, int):
                print(f"Warning: Scene index at position {idx} is not an integer: {scene_id}")
                return False
        return True

    # Direct parse
    try:
        result = json.loads(output)
        if _validate(result):
            return result
    except Exception:
        pass

    # Strip ```json ... ``` fences
    m = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL | re.IGNORECASE).search(output)
    if m:
        try:
            result = json.loads(m.group(1))
            if _validate(result):
                return result
        except Exception:
            pass

    # Find first '{' or '['
    json_start = min((i for i in (output.find("{"), output.find("[")) if i != -1), default=None)
    if json_start is not None:
        try:
            result = json.loads(output[json_start:])
            if _validate(result):
                return result
        except Exception:
            pass

    # Last resort: any {...} block
    for b in re.findall(r'({.*})', output, re.DOTALL):
        try:
            result = json.loads(b)
            if _validate(result):
                return result
        except Exception:
            continue

    print("parse_structure_proposal_output: all attempts failed.")
    print(output[:500])
    return None


def parse_shot_plan_output(output: str) -> Optional[Dict]:
    """Parse shot plan JSON from LLM output with light recovery for mixed text."""
    if not output:
        return None
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    candidates = [text]

    json_start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=None)
    if json_start is not None:
        candidate = text[json_start:].strip()
        if candidate not in candidates:
            candidates.append(candidate)

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1).strip()
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None
