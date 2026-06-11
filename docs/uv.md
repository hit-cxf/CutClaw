# UV setup

CutClaw now uses `uv` for the default macOS-friendly environment.

## Install the default environment

```bash
uv sync
```

If you want to create the lockfile separately first:

```bash
uv lock
```

Run the UI:

```bash
uv run streamlit run app.py
```

Run from the command line:

```bash
uv run python local_run.py \
  --Video_Path resource/video/example.mp4 \
  --Audio_Path resource/audio/example.mp3 \
  --Instruction "Describe the edit you want"
```

Render an existing `shot_point` file:

```bash
uv run python render/render_video.py \
  --shot-plan Output/Output/<video>_<audio>/shot_plan_xxx.json \
  --shot-json Output/Output/<video>_<audio>/shot_point_xxx.json \
  --video resource/video/example.mp4 \
  --audio resource/audio/example.mp3 \
  --output output.mp4
```

## Optional extras

The default install avoids legacy or platform-fragile packages.

```bash
uv sync --extra decord
uv sync --extra local-asr
uv sync --extra diarization
uv sync --extra qwen
```

`madmom` and `aubio` are intentionally not managed by the default uv
environment. They are legacy compiled packages and are the main source of
macOS/Python 3.12 installation failures. If you need the original madmom
detector stack, use a separate Linux/Python 3.10 environment and install those
packages manually.

On macOS, CutClaw uses the PyAV video reader fallback when `decord` is not
available. You still need the `ffmpeg` executable in `PATH` for audio conversion
and final rendering.
