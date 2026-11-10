"""Opt-in, per-request LiteLLM accounting for benchmark runs."""

import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import Lock
from uuid import uuid4

import litellm


def token_count(data, key):
    value = data.get(key)
    return value if isinstance(value, int) and value >= 0 else None


def normalized_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    usage = usage.model_dump(exclude_none=True)
    inputs = usage.get("prompt_tokens_details", {})
    outputs = usage.get("completion_tokens_details", {})
    return {
        "input_tokens": token_count(usage, "prompt_tokens"),
        "output_tokens": token_count(usage, "completion_tokens"),
        "total_tokens": token_count(usage, "total_tokens"),
        "cached_input_tokens": token_count(inputs, "cached_tokens"),
        "audio_input_tokens": token_count(inputs, "audio_tokens"),
        "audio_output_tokens": token_count(outputs, "audio_tokens"),
    }


class ModelUsageLogger:
    def __init__(self, path, prices):
        self.path = path
        self.prices = prices
        self.attempt_id = uuid4().hex
        self.phase = "material_analysis"
        self.lock = Lock()

    def _cost(self, model, usage):
        prices = self.prices.get(model)
        prompt = usage.get("input_tokens")
        output = usage.get("output_tokens")
        if not prices or prompt is None or output is None:
            return None
        if prompt > prices.get("max_input_tokens", float("inf")):
            return None
        if "audio_input" in prices:
            audio = usage.get("audio_input_tokens")
            if audio is None or not 0 <= audio <= prompt:
                return None
            if usage.get("audio_output_tokens"):
                return None
            return ((prompt - audio) * prices["input"] + audio * prices["audio_input"]
                    + output * prices["output"]) / 1_000_000
        cached = usage.get("cached_input_tokens") or 0
        if cached > prompt:
            return None
        return ((prompt - cached) * prices["input"] + cached * prices["cached_input"]
                + output * prices["output"]) / 1_000_000

    def _write(self, event, request, response=None):
        usage = normalized_usage(response)
        row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "attempt_id": self.attempt_id,
            "event": event,
            **request,
            **usage,
            "estimated_cost_cny": self._cost(request["model"], usage),
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")

    def _start(self, args, kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        request = {
            "request_id": uuid4().hex,
            "phase": self.phase,
            "model": model.rsplit("/", 1)[-1],
        }
        self._write("started", request)
        return request

    def _stream(self, response, request):
        usage_chunk = None
        try:
            for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage_chunk = chunk
                yield chunk
        except Exception:
            self._write("failed", request, usage_chunk)
            raise
        self._write("succeeded", request, usage_chunk)

    async def _async_stream(self, response, request):
        usage_chunk = None
        try:
            async for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    usage_chunk = chunk
                yield chunk
        except Exception:
            self._write("failed", request, usage_chunk)
            raise
        self._write("succeeded", request, usage_chunk)

    def install(self):
        complete = litellm.completion
        async_complete = litellm.acompletion

        @wraps(complete)
        def tracked_completion(*args, **kwargs):
            request = self._start(args, kwargs)
            try:
                response = complete(*args, **kwargs)
            except Exception:
                self._write("failed", request)
                raise
            if kwargs.get("stream"):
                return self._stream(response, request)
            self._write("succeeded", request, response)
            return response

        @wraps(async_complete)
        async def tracked_acompletion(*args, **kwargs):
            request = self._start(args, kwargs)
            try:
                response = await async_complete(*args, **kwargs)
            except Exception:
                self._write("failed", request)
                raise
            if kwargs.get("stream"):
                return self._async_stream(response, request)
            self._write("succeeded", request, response)
            return response

        litellm.completion = tracked_completion
        litellm.acompletion = tracked_acompletion


def configure_model_usage():
    destination = os.environ.get("CUTCLAW_MODEL_USAGE_PATH")
    if not destination:
        return None
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    price_file = os.environ.get("CUTCLAW_MODEL_PRICES_PATH")
    prices = json.loads(Path(price_file).read_text()) if price_file else {}
    logger = ModelUsageLogger(path, prices)
    logger.install()
    return logger
