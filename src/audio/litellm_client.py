"""
Lightweight litellm wrapper for audio analysis via cloud API.

Reads configuration from src/audio/.env:
  AUDIO_MODEL    - LiteLLM model string (e.g. dashscope/qwen3.5-omni-plus)
  AUDIO_API_KEY  - API key (use EMPTY for no auth)
  AUDIO_BASE_URL - Base URL for OpenAI-compatible endpoints
"""

import os
import asyncio
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import List

import litellm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from .. import config as project_config
except Exception:
    project_config = None

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")


def _get_setting(config_key: str, env_key: str, default=None):
    """Read setting from src/config.py first, then fallback to environment/.env."""
    if project_config is not None and hasattr(project_config, config_key):
        value = getattr(project_config, config_key)
        if value is not None and value != "":
            return value
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


AUDIO_MODEL = _get_setting(
    config_key="AUDIO_LITELLM_MODEL",
    env_key="AUDIO_MODEL",
    default="dashscope/qwen3.5-omni-plus",
)
AUDIO_API_KEY = _get_setting(
    config_key="AUDIO_LITELLM_API_KEY",
    env_key="AUDIO_API_KEY",
    default="EMPTY",
) or "EMPTY"
AUDIO_BASE_URL = _get_setting(
    config_key="AUDIO_LITELLM_BASE_URL",
    env_key="AUDIO_BASE_URL",
    default=None,
)


def _audio_to_base64_mp3(audio_path: str) -> str:
    """Convert audio file to base64-encoded MP3 for cloud API submission."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", "-ab", "32k", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _is_qwen_omni_model(model: str) -> bool:
    """Return True for Qwen Omni models that require streaming audio calls."""
    model_name = (model or "").split("/")[-1].lower()
    return "qwen" in model_name and "omni" in model_name


async def _collect_stream_text(stream) -> str:
    """Collect text deltas from LiteLLM/OpenAI-compatible streaming chunks."""
    chunks = []
    async for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content:
            chunks.append(content)
    return "".join(chunks)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),  # 1 次正常 + 最多 3 次重试
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: print(
        f"[litellm_client] Retry {rs.attempt_number}/3 "
        f"after {rs.outcome.exception()} — sleeping {rs.next_action.sleep:.1f}s"
    ),
)
async def acall_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Async: Call the cloud audio API with a prompt about the given audio file.

    Audio is converted to MP3 once (not retried); the API call is retried up to 3
    times with exponential backoff (1s → 2s → 4s, capped at 30s) on any error.

    Args:
        audio_path: Path to audio file
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate

    Returns:
        Response text from the model
    """
    loop = asyncio.get_running_loop()
    # ffmpeg conversion only runs once — not included in the retry scope
    audio_b64 = await loop.run_in_executor(None, _audio_to_base64_mp3, audio_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/mp3;base64,{audio_b64}",
                    "format": "mp3",
                },
            },
        ],
    }]

    kwargs = {
        "model": AUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "timeout": 300,
        "api_key": AUDIO_API_KEY,
        **({"api_base": AUDIO_BASE_URL} if AUDIO_BASE_URL else {}),
    }

    if _is_qwen_omni_model(AUDIO_MODEL):
        response_stream = await litellm.acompletion(
            **kwargs,
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )
        return await _collect_stream_text(response_stream)

    response = await litellm.acompletion(**kwargs)

    return response.choices[0].message.content


def call_audio_api(
    audio_path: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Sync wrapper around acall_audio_api.

    Args:
        audio_path: Path to audio file
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate

    Returns:
        Response text from the model
    """
    return asyncio.run(acall_audio_api(audio_path, prompt, temperature, top_p, max_tokens))


async def acall_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_concurrent: int = 5,
) -> List[str]:
    """
    Async: Call the cloud audio API concurrently for multiple audio files.

    Uses asyncio.gather with a semaphore to limit concurrent requests.
    Each request is retried automatically via the @retry decorator on acall_audio_api.

    Args:
        audio_paths: List of paths to audio files
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
        max_concurrent: Max simultaneous API requests

    Returns:
        List of response texts (same order as audio_paths, "" on error)
    """
    if not audio_paths:
        return []

    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(path: str) -> str:
        async with sem:
            return await acall_audio_api(path, prompt, temperature, top_p, max_tokens)

    results = await asyncio.gather(
        *[_limited(p) for p in audio_paths],
        return_exceptions=True,
    )

    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[litellm_client] Failed after retries: {audio_paths[i]}: {result}")
            processed.append("")
        else:
            processed.append(result)
    return processed


def call_audio_api_batch(
    audio_paths: List[str],
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    max_workers: int = 5,
) -> List[str]:
    """
    Sync wrapper around acall_audio_api_batch.

    Args:
        audio_paths: List of paths to audio files
        prompt: Text prompt for analysis
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
        max_workers: Max simultaneous API requests

    Returns:
        List of response texts (same order as audio_paths, "" on error)
    """
    return asyncio.run(
        acall_audio_api_batch(audio_paths, prompt, temperature, top_p, max_tokens, max_workers)
    )
