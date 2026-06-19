"""
场景级视频 Caption 脚本
直接使用已提取的帧进行场景理解，生成以人物为主体的连贯叙事描述
"""
import os
import sys
import json
import re
import asyncio
import threading
import traceback
from typing import List, Dict, Optional

import litellm
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.prompt import SCENE_VIDEO_CAPTION_PROMPT, VLOG_SCENE_CAPTION_PROMPT
from src.video.preprocess.video_utils import load_cached_sampled_frames
from src.utils.media_utils import (
    parse_json_safely,
    parse_srt_file,
    get_subtitles_in_range,
    format_subtitles,
    hhmmss_to_seconds,
    pil_to_base64,
)


def _is_valid_scene_analysis_output(output_path: str) -> bool:
    """Check whether an existing scene analysis output is complete enough to skip."""
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        video_analysis = data.get('video_analysis', {})
        scene_caption = video_analysis.get('scene_caption')
        if not isinstance(scene_caption, dict):
            return False
        scene_classification = scene_caption.get('scene_classification', {})
        return isinstance(scene_classification, dict)
    except Exception:
        return False


def load_scene_frames_from_vr(video_reader, frame_indices: List[int], frame_range: List[int],
                               frame_paths: Optional[List[str]] = None,
                               max_frames: int = None,
                               min_frames: int = None) -> List[Image.Image]:
    """
    从 decord VideoReader 加载场景帧（内存读取，无磁盘 I/O）
    frame_range: [start_frame, end_frame]，单位为采样帧空间索引
    frame_indices: decode_video_to_frames 返回的原始视频帧索引列表
    """
    if max_frames is None:
        max_frames = config.CAPTION_BATCH_SIZE
    if min_frames is None:
        min_frames = config.SCENE_ANALYSIS_MIN_FRAMES
    start_frame, end_frame = frame_range
    # 取出该场景在采样帧空间内的所有索引
    scene_sampled = list(range(start_frame, end_frame + 1))
    total_scene_frames = len(scene_sampled)

    if total_scene_frames == 0:
        return []

    # 计算采样数量
    num_samples = min(max(min_frames, total_scene_frames), max_frames)
    num_samples = min(num_samples, total_scene_frames)

    # 均匀采样
    if num_samples >= total_scene_frames:
        sampled_scene_indices = scene_sampled
    else:
        step = total_scene_frames / num_samples
        sampled_scene_indices = [scene_sampled[int(i * step)] for i in range(num_samples)]

    if frame_paths and len(frame_paths) == len(frame_indices):
        arrays = load_cached_sampled_frames(
            frame_paths=frame_paths,
            sampled_indices=sampled_scene_indices,
            target_short_side=int(getattr(config, "VIDEO_RESOLUTION", 0) or 0),
        )
        if arrays:
            return [Image.fromarray(arr) for arr in arrays]

    # 将采样帧空间索引映射到原始视频帧索引
    orig_indices = [frame_indices[i] for i in sampled_scene_indices if i < len(frame_indices)]
    if not orig_indices:
        return []

    try:
        batch = video_reader.get_batch(orig_indices).asnumpy()  # (N, H, W, C)
        return [Image.fromarray(batch[i]) for i in range(len(orig_indices))]
    except Exception as e:
        print(f"⚠️  [SceneAnalysis] Warning: Failed to load frames from video_reader: {e}")
        return []



def extract_known_characters(shots_data: List[Dict]) -> str:
    """从镜头数据提取已知人物"""
    chars = {}
    for shot in shots_data:
        for c in shot.get('entities', {}).get('active_characters', []):
            vid = c.get('visual_id', 'Unknown')
            if vid not in chars:
                chars[vid] = c.get('appearance', 'No description')

    if not chars:
        return "No prior character information."

    return "\n".join([f"- {vid}: {desc}" for vid, desc in chars.items()])



class SceneVideoAnalyzer:
    def __init__(self, vr: Dict, subtitle_file: Optional[str] = None):
        """
        Args:
            vr: Dict returned by decode_video_to_frames, containing
                'video_reader' (decord VideoReader) and 'frame_indices' (List[int])
            subtitle_file: Optional path to .srt subtitle file
        """
        self.video_reader = vr["video_reader"]
        self.frame_indices = vr["frame_indices"]
        self.frame_paths = vr.get("frame_paths") or []
        self.subtitles = parse_srt_file(subtitle_file) if subtitle_file else []
        if self.subtitles:
            print(f"✅ [SceneAnalysis] Loaded {len(self.subtitles)} subtitle entries")

    async def _call_vlm(self, system_prompt: str, content: List[Dict], max_tokens: int = 4096, timeout: float = 120) -> Optional[str]:
        """调用 VLM"""
        kwargs = dict(
            model=config.VIDEO_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if config.VIDEO_ANALYSIS_ENDPOINT:
            kwargs["api_base"] = config.VIDEO_ANALYSIS_ENDPOINT
        if config.VIDEO_ANALYSIS_API_KEY:
            kwargs["api_key"] = config.VIDEO_ANALYSIS_API_KEY

        try:
            response = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=timeout)
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ [SceneAnalysis] VLM Error: {type(e).__name__}: {e}")
            return None

    def _build_content(self, frames: List[Image.Image], text_parts: List[str]) -> List[Dict]:
        """构建 VLM 输入"""
        content = [{"type": "text", "text": "\n".join(text_parts)}]

        content.append({"type": "text", "text": f"\n=== Scene Frames ({len(frames)} frames) ==="})
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(frame)}"}
            })

        return content

    async def generate_caption(self, frames: List[Image.Image], dialogue: str, known_chars: str, max_retries: int = 5) -> Optional[Dict]:
        """生成场景描述，验证必需字段"""
        # 根据配置选择 prompt
        base_prompt = VLOG_SCENE_CAPTION_PROMPT if config.SCENE_PROMPT_TYPE == "vlog" else SCENE_VIDEO_CAPTION_PROMPT

        # 格式化人物/位置信息
        if config.SCENE_PROMPT_TYPE == "vlog":
            location_text = "Unknown location"
            prompt = base_prompt.replace("{LOCATION}", location_text).replace("{DIALOGUE}", dialogue)
            content = self._build_content(frames, [f"Location:\n{location_text}", f"\nCreator Speech:\n{dialogue}"])
        else:
            char_text = known_chars if known_chars and known_chars.strip() else "No characters identified."
            prompt = base_prompt.replace("{CHARACTERS}", char_text).replace("{DIALOGUE}", dialogue)
            content = self._build_content(frames, [f"Characters:\n{char_text}", f"\nDialogue:\n{dialogue}"])

        timeouts = [80, 120, 120, 180, 180]
        for attempt in range(max_retries):
            timeout = timeouts[min(attempt, len(timeouts) - 1)]
            result = await self._call_vlm(prompt, content, max_tokens=4096, timeout=timeout)
            parsed = parse_json_safely(result) if result else None

            # 验证必需字段
            if parsed and 'scene_classification' in parsed:
                classification = parsed['scene_classification']
                # 确保有 importance_score
                if 'importance_score' not in classification:
                    scene_type = classification.get('scene_type', 'content')
                    film_unusable_types = ['studio_logo', 'title_card', 'credits', 'transition']
                    vlog_unusable_types = ['transition', 'technical_issue']

                    if config.SCENE_PROMPT_TYPE == "vlog":
                        if scene_type in vlog_unusable_types:
                            classification['importance_score'] = 0
                            classification['is_usable'] = False
                        elif scene_type == 'mixed':
                            classification['importance_score'] = 2
                            classification['is_usable'] = True
                        else:
                            classification['importance_score'] = 3
                            classification['is_usable'] = True
                    else:
                        if scene_type in film_unusable_types:
                            classification['importance_score'] = 0
                            classification['is_usable'] = False
                        elif scene_type == 'mixed':
                            classification['importance_score'] = 2
                            classification['is_usable'] = True
                        else:
                            classification['importance_score'] = 3
                            classification['is_usable'] = True

                if classification.get('importance_score', 0) == 0:
                    classification['is_usable'] = False

                return parsed

            if attempt < max_retries - 1:
                print(f"⚠️  [SceneAnalysis] Output missing required fields, retrying ({attempt + 1}/{max_retries})...")

        print(f"❌ [SceneAnalysis] Failed to get valid output after {max_retries} attempts")
        return parsed

    def load_scene_frames(self, scene_data: Dict) -> List:
        """同步读帧（decord not thread-safe，必须在 event loop 主线程调用）"""
        if 'frame_range' in scene_data:
            frame_range = scene_data['frame_range']
        elif 'start_frame' in scene_data and 'end_frame' in scene_data:
            frame_range = [scene_data['start_frame'], scene_data['end_frame']]
        else:
            frame_range = [0, 0]
        return load_scene_frames_from_vr(self.video_reader, self.frame_indices, frame_range, frame_paths=self.frame_paths)

    async def process_scene(self, scene_data: Dict, frames: List = None) -> Dict:
        """处理单个场景，frames 可由外部预先读取以避免阻塞 event loop"""
        if 'frame_range' in scene_data:
            frame_range = scene_data['frame_range']
        elif 'start_frame' in scene_data and 'end_frame' in scene_data:
            frame_range = [scene_data['start_frame'], scene_data['end_frame']]
        else:
            frame_range = [0, 0]

        if frames is None:
            # fallback: 直接读帧（会阻塞 event loop，仅兼容旧调用）
            frames = load_scene_frames_from_vr(self.video_reader, self.frame_indices, frame_range, frame_paths=self.frame_paths)
        if not frames:
            return {"error": "No frames loaded"}

        # 获取时间范围和字幕 - 兼容两种格式
        # 格式1: time_range = {start_seconds: "HH:MM:SS", end_seconds: "HH:MM:SS"}  (场景合并后的格式)
        # 格式2: start_time, end_time (秒数)  (固定长度场景的格式)
        if 'time_range' in scene_data:
            time_range = scene_data['time_range']
            start_sec = hhmmss_to_seconds(time_range.get('start_seconds', '00:00:00'))
            end_sec = hhmmss_to_seconds(time_range.get('end_seconds', '00:00:00'))
        elif 'start_time' in scene_data and 'end_time' in scene_data:
            start_sec = float(scene_data['start_time'])
            end_sec = float(scene_data['end_time'])
        else:
            start_sec = 0.0
            end_sec = 0.0

        scene_subs = get_subtitles_in_range(self.subtitles, start_sec, end_sec)
        dialogue = format_subtitles(scene_subs)

        # 获取已知人物
        known_chars = extract_known_characters(scene_data.get('shots_data', []))

        # 直接使用已有人物信息生成场景描述，避免重复人物识别调用
        caption = await self.generate_caption(frames, dialogue, known_chars)

        return {
            'character_identification': None,
            'scene_caption': caption,
            'dialogue': scene_subs,
            'frames_used': len(frames),
            'frame_range': frame_range
        }

    async def process_file(self, in_path: str, out_path: str, frames: List = None, scene_data: Dict = None) -> str:
        """处理单个文件，scene_data/frames 可由外部预先提供"""
        if _is_valid_scene_analysis_output(out_path):
            return "Skipped"

        try:
            if os.path.exists(out_path):
                print(f"⚠️  [SceneAnalysis] Found invalid/incomplete output, regenerating: {out_path}")

            if scene_data is None:
                with open(in_path, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)

            result = await self.process_scene(scene_data, frames=frames)
            scene_data['video_analysis'] = result

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)

            return "Success"
        except Exception as e:
            return f"Error: {e}"

    def analyze_scenes_dir(
        self,
        scenes_dir: str,
        output_dir: str,
        max_workers: int = 8,
        overwrite: bool = False,
    ) -> Dict:
        """批量分析场景目录并返回统计信息。"""
        if not os.path.exists(scenes_dir):
            return {
                "status": "invalid",
                "total_scenes": 0,
                "already_analyzed": 0,
                "success": 0,
                "skipped": 0,
                "errors": [f"Scenes directory not found: {scenes_dir}"],
            }

        scene_files = sorted(
            [f for f in os.listdir(scenes_dir) if f.endswith('.json')],
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]
        )
        if not scene_files:
            return {
                "status": "skipped",
                "total_scenes": 0,
                "already_analyzed": 0,
                "success": 0,
                "skipped": 0,
                "errors": [],
            }

        os.makedirs(output_dir, exist_ok=True)

        tasks = []
        already_analyzed = 0
        for file_name in scene_files:
            in_path = os.path.join(scenes_dir, file_name)
            out_path = os.path.join(output_dir, file_name)
            if (not overwrite) and _is_valid_scene_analysis_output(out_path):
                already_analyzed += 1
                continue
            tasks.append((in_path, out_path))

        if not tasks:
            return {
                "status": "skipped",
                "total_scenes": len(scene_files),
                "already_analyzed": already_analyzed,
                "success": 0,
                "skipped": already_analyzed,
                "errors": [],
            }

        async def _caption_one(in_path, out_path, scene_data, frames, semaphore):
            async with semaphore:
                try:
                    return await self.process_file(in_path, out_path, scene_data=scene_data, frames=frames)
                except Exception as e:
                    return f"Error: {e}"

        async def _run_all(task_list, pbar):
            semaphore = asyncio.Semaphore(max_workers)
            failed_tasks = []
            queue_size = max(1, max_workers * 2)
            queue = asyncio.Queue(maxsize=queue_size)
            sentinel = object()
            loop = asyncio.get_running_loop()
            producer_errors = []

            def _producer():
                try:
                    for in_path, out_path in task_list:
                        with open(in_path, 'r', encoding='utf-8') as f:
                            scene_data = json.load(f)
                        frames = self.load_scene_frames(scene_data)
                        future = asyncio.run_coroutine_threadsafe(
                            queue.put((in_path, out_path, scene_data, frames)),
                            loop,
                        )
                        future.result()
                except Exception as exc:
                    producer_errors.append(exc)
                finally:
                    for _ in range(max_workers):
                        future = asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)
                        future.result()

            async def _consumer():
                while True:
                    item = await queue.get()
                    try:
                        if item is sentinel:
                            return

                        in_path, out_path, scene_data, frames = item
                        result = await _caption_one(
                            in_path,
                            out_path,
                            scene_data,
                            frames,
                            semaphore,
                        )
                        if result == "Success":
                            pbar.update(1)
                        else:
                            failed_tasks.append((in_path, out_path))
                    except Exception:
                        scene_name = os.path.basename(item[0]) if isinstance(item, tuple) and item else "<unknown>"
                        print(f"❌ [SceneAnalysis] Consumer error while processing {scene_name}:")
                        traceback.print_exc()
                        raise
                    finally:
                        queue.task_done()

            producer = threading.Thread(
                target=_producer,
                name="scene-analysis-frame-reader",
                daemon=True,
            )
            producer.start()
            consumers = [asyncio.create_task(_consumer()) for _ in range(max_workers)]

            await asyncio.gather(*consumers)
            producer.join()

            if producer_errors:
                raise producer_errors[0]

            return failed_tasks

        pending_tasks = list(tasks)
        pbar = tqdm(total=len(pending_tasks), desc="Analyzing scenes")
        all_results = []

        loop = asyncio.new_event_loop()
        try:
            failed = loop.run_until_complete(_run_all(pending_tasks, pbar))
        finally:
            loop.close()
        succeeded_count = len(pending_tasks) - len(failed)
        all_results.extend(["Success"] * succeeded_count)

        if failed:
            print(f"  ❌ [SceneAnalysis] Warning: {len(failed)} scenes still failed after all retries exhausted internally")

        pbar.close()
        results = all_results

        success_count = results.count("Success")
        skipped_count = results.count("Skipped") + already_analyzed
        errors = [r for r in results if isinstance(r, str) and r.startswith("Error")]

        return {
            "status": "completed",
            "total_scenes": len(scene_files),
            "already_analyzed": already_analyzed,
            "success": success_count,
            "skipped": skipped_count,
            "errors": errors,
        }


# ================= 主程序 =================

def main():
    import argparse
    from src.video.preprocess import decode_video_to_frames

    parser = argparse.ArgumentParser(description="Scene video analysis")
    parser.add_argument("--video_path", required=True, help="Path to source video file")
    parser.add_argument("--scenes_dir", required=True, help="Path to scenes JSON directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--subtitle_file", default=None, help="Optional path to .srt subtitle file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    vr = decode_video_to_frames(
        args.video_path,
        frames_dir=os.path.join(os.path.dirname(args.output_dir), "frames"),
        target_fps=config.VIDEO_FPS,
        target_resolution=config.VIDEO_RESOLUTION,
    )

    print("🔍 [SceneAnalysis] Scene Video Analysis")
    print(f"  📂 Scenes: {args.scenes_dir}")
    print(f"  🤖 Model: {config.VIDEO_ANALYSIS_MODEL.split('/', 1)[-1]}")
    print(f"  Prompt type: {config.SCENE_PROMPT_TYPE} ({'Travel Vlog' if config.SCENE_PROMPT_TYPE == 'vlog' else 'Film/TV'})")
    print(f"  🎞️  Max frames/scene: {config.CAPTION_BATCH_SIZE}")

    analyzer = SceneVideoAnalyzer(vr=vr, subtitle_file=args.subtitle_file)

    stats = analyzer.analyze_scenes_dir(
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        max_workers=config.CAPTION_BATCH_SIZE,
        overwrite=args.overwrite,
    )

    print(
        f"\n✅ [SceneAnalysis] Done: {stats.get('success', 0)} success, "
        f"{stats.get('skipped', 0)} skipped"
    )
    errors = stats.get("errors", [])
    if errors:
        print(f"❌ [SceneAnalysis] Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
