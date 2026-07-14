# Local TTS Backend (sherpa-onnx) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 6th offline TTS backend (`--provider sherpa`, based on sherpa-onnx) to `notes_to_audio.py`, enabling fully-offline Chinese narration synthesis after a one-time manual model setup.

**Architecture:** New `backend_sherpa.py` module follows the existing plugin-dispatch pattern (same contract as the other 5 backends). `notes_to_audio.py` gains 7 new dispatch branches — zero changes to existing providers. The model is supplied by the user via `SHERPA_TTS_MODEL_DIR` (no auto-download; zero network calls in code).

**Tech Stack:** Python 3, sherpa-onnx (VITS inference, CPU), stdlib `wave` for int16 WAV output.

**Testing convention:** Per `docs/rules/code-style.md §11`, this repo ships **no automated tests** (no `tests/`, no `pytest`). Every task below verifies with manual commands, not test files. Do NOT add `test_*.py` or `pytest` imports.

---

## Task 1: Create `backend_sherpa.py` — contract layer

**Files:**
- Create: `skills/ppt-master/scripts/tts_backends/backend_sherpa.py`

This task adds the no-dependency contract functions. They import successfully even before sherpa-onnx is installed, so verification needs no model and no network.

- [ ] **Step 1: Create the module with contract functions**

Create `skills/ppt-master/scripts/tts_backends/backend_sherpa.py`:

```python
"""Local offline TTS backend (sherpa-onnx) for narration audio.

Pure-offline synthesis: the user prepares a sherpa-onnx VITS model once and
points SHERPA_TTS_MODEL_DIR at it. This module makes zero network calls.
"""

from __future__ import annotations

import os
from pathlib import Path


def output_extension() -> str:
    """Audio extension produced by this backend (int16 WAV)."""
    return ".wav"


def read_sherpa_config() -> dict:
    """Read model config from the environment. Fail fast if unconfigured."""
    model_dir = os.environ.get("SHERPA_TTS_MODEL_DIR", "").strip()
    if not model_dir:
        raise RuntimeError(
            "Set SHERPA_TTS_MODEL_DIR=<sherpa-onnx 模型目录>"
            "（手动下载模型后解压到此目录，代码不做自动下载）"
        )
    path = Path(model_dir)
    if not path.is_dir():
        raise RuntimeError(f"SHERPA_TTS_MODEL_DIR 不是目录: {model_dir}")
    return {"model_dir": path}


def print_voices() -> None:
    """Print the local speaker catalog. Offline models have no online list."""
    print("sherpa-onnx speakers (本地模型 speaker id):")
    print("ID   Note")
    print("0    默认说话人（单说话人模型唯一可选）")
    print("0..N 多说话人模型(如 aishell3)可选 0~N，取决于模型")
```

- [ ] **Step 2: Verify contract functions**

Run from repo root:
```bash
cd skills/ppt-master/scripts && python3 -c "
from tts_backends import backend_sherpa
print('ext:', backend_sherpa.output_extension())
backend_sherpa.print_voices()
try:
    backend_sherpa.read_sherpa_config()
    print('FAIL: should have raised')
except RuntimeError as e:
    print('OK raised:', str(e)[:40])
"
```
Expected output:
```
ext: .wav
sherpa-onnx speakers (本地模型 speaker id):
ID   Note
0    默认说话人（单说话人模型唯一可选）
0..N 多说话人模型(如 aishell3)可选 0~N，取决于模型
OK raised: Set SHERPA_TTS_MODEL_DIR=<sherpa-onnx 模型目录>
```

- [ ] **Step 3: Commit**
```bash
git add skills/ppt-master/scripts/tts_backends/backend_sherpa.py
git commit -m "feat(audio): add backend_sherpa contract layer (offline TTS)"
```

---

## Task 2: Add synthesis layer to `backend_sherpa.py`

**Files:**
- Modify: `skills/ppt-master/scripts/tts_backends/backend_sherpa.py`

This adds the synthesis functions. They import `sherpa_onnx` lazily (inside functions) so the module still imports without the dependency.

- [ ] **Step 1: Install sherpa-onnx**
```bash
python3 -m pip install sherpa-onnx
```
Verify: `python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)"` prints a version.

- [ ] **Step 2: Append synthesis functions**

Append to `skills/ppt-master/scripts/tts_backends/backend_sherpa.py` (after `print_voices`):

```python
def _build_config(model_dir: Path):
    """Build OfflineTtsConfig by probing the model dir for known files.

    Probes model.onnx / model.int8.onnx, tokens.txt, and optional lexicon.txt
    so the same code works across single/multi-speaker Chinese VITS models.
    """
    import sherpa_onnx

    model_file = model_dir / "model.onnx"
    if not model_file.exists():
        model_file = model_dir / "model.int8.onnx"
    if not model_file.exists():
        raise RuntimeError(
            f"未在 {model_dir} 找到 model.onnx 或 model.int8.onnx"
        )

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise RuntimeError(f"未在 {model_dir} 找到 tokens.txt")

    lexicon = model_dir / "lexicon.txt"
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model_file),
        tokens=str(tokens),
        lexicon=str(lexicon) if lexicon.exists() else "",
    )
    return sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            num_threads=1,
            debug=False,
            provider="cpu",
        ),
        max_num_sentences=2,
    )


def _get_tts():
    """Lazily load OfflineTts and cache it as a function attribute.

    The notes loop calls generate() per slide; without this cache each slide
    would reload the model.
    """
    if not hasattr(_get_tts, "_tts"):
        import sherpa_onnx  # noqa: F401  (config builder imports it too)
        cfg = read_sherpa_config()
        _get_tts._tts = sherpa_onnx.OfflineTts(_build_config(cfg["model_dir"]))
    return _get_tts._tts


def _write_wav(output_path: Path, samples, sample_rate: int) -> None:
    """Write float32 samples as a mono int16 WAV using only the stdlib."""
    import struct
    import wave

    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            value = int(max(-1.0, min(1.0, float(sample))) * 32767)
            frames += struct.pack("<h", value)
        writer.writeframes(bytes(frames))


def generate(
    text: str,
    output_path: Path,
    *,
    voice_id: str = "0",
    speed: float = 1.0,
    **_,
) -> None:
    """Synthesize one note into a WAV file."""
    tts = _get_tts()
    sid = int(voice_id) if str(voice_id).isdigit() else 0
    audio = tts.generate(text, sid=sid, speed=float(speed))
    _write_wav(output_path, audio.samples, audio.sample_rate)
```

- [ ] **Step 3: Verify the module still imports and field names resolve**

Run:
```bash
cd skills/ppt-master/scripts && python3 -c "
from tts_backends import backend_sherpa
# Confirm the OfflineTtsVitsModelConfig accepts the fields we use.
import sherpa_onnx
v = sherpa_onnx.OfflineTtsVitsModelConfig(model='x', tokens='y', lexicon='')
print('fields OK; has generate:', hasattr(backend_sherpa, 'generate'))
"
```
Expected: `fields OK; has generate: True`

> If `OfflineTtsVitsModelConfig` rejects a field name, run
> `python3 -c "import sherpa_onnx; help(sherpa_onnx.OfflineTtsVitsModelConfig)"`
> and adjust the three field names in `_build_config` to match the installed
> version. This is the only sherpa-onnx-version-sensitive line in the plan.

- [ ] **Step 4: Commit**
```bash
git add skills/ppt-master/scripts/tts_backends/backend_sherpa.py
git commit -m "feat(audio): add backend_sherpa synthesis layer (VITS generate)"
```

---

## Task 3: Integrate sherpa into `notes_to_audio.py` — static wiring

**Files:**
- Modify: `skills/ppt-master/scripts/notes_to_audio.py`

Adds the import, env-file prefix, the new `--sherpa-speed` flag, and the `--list-voices` dispatch. These are verifiable without a model.

- [ ] **Step 1: Add the import**

In `skills/ppt-master/scripts/notes_to_audio.py`, the existing import block (around line 35) is:
```python
from tts_backends import (
    backend_cosyvoice,
    backend_edge,
    backend_elevenlabs,
    backend_minimax,
    backend_qwen,
)
```
Change it to:
```python
from tts_backends import (
    backend_cosyvoice,
    backend_edge,
    backend_elevenlabs,
    backend_minimax,
    backend_qwen,
    backend_sherpa,
)
```

- [ ] **Step 2: Add the `SHERPA_TTS_` env prefix**

In `_load_tts_env_file` (around line 54), change:
```python
def _load_tts_env_file() -> None:
    """Load TTS-related keys from the first .env file, without overriding shell env."""
    load_prefixed_env_file((
        "ELEVENLABS_",
        "MINIMAX_",
        "QWEN_",
        "DASHSCOPE_",
        "COSYVOICE_",
    ))
```
to:
```python
def _load_tts_env_file() -> None:
    """Load TTS-related keys from the first .env file, without overriding shell env."""
    load_prefixed_env_file((
        "ELEVENLABS_",
        "MINIMAX_",
        "QWEN_",
        "DASHSCOPE_",
        "COSYVOICE_",
        "SHERPA_TTS_",
    ))
```

- [ ] **Step 3: Add the `--sherpa-speed` argument**

Find the cosyvoice argument block ending around line 202 (the last `--cosyvoice-*` argument). Immediately after it, add:
```python
    parser.add_argument(
        "--sherpa-speed",
        type=float,
        default=1.0,
        help="sherpa-onnx speaking speed multiplier (default: 1.0)",
    )
```

- [ ] **Step 4: Add the `--list-voices` dispatch branch**

In the `if args.list_voices:` block (around line 214-224), the existing dispatch ends with:
```python
            elif args.provider == "cosyvoice":
                backend_cosyvoice.print_voices()
            else:
                asyncio.run(backend_edge.print_voices(args.locale))
```
Change it to insert a sherpa branch before `else`:
```python
            elif args.provider == "cosyvoice":
                backend_cosyvoice.print_voices()
            elif args.provider == "sherpa":
                backend_sherpa.print_voices()
            else:
                asyncio.run(backend_edge.print_voices(args.locale))
```

- [ ] **Step 5: Verify static wiring**

Run:
```bash
cd skills/ppt-master/scripts && python3 notes_to_audio.py --provider sherpa --list-voices
```
Expected output:
```
sherpa-onnx speakers (本地模型 speaker id):
ID   Note
0    默认说话人（单说话人模型唯一可选）
0..N 多说话人模型(如 aishell3)可选 0~N，取决于模型
```
Also confirm the flag is registered:
```bash
cd skills/ppt-master/scripts && python3 notes_to_audio.py --help | grep sherpa-speed
```
Expected: a line showing `--sherpa-speed SHERPA_SPEED`.

- [ ] **Step 6: Commit**
```bash
git add skills/ppt-master/scripts/notes_to_audio.py
git commit -m "feat(audio): wire sherpa provider import, env, flag, list-voices"
```

---

## Task 4: Integrate sherpa into `notes_to_audio.py` — runtime dispatch

**Files:**
- Modify: `skills/ppt-master/scripts/notes_to_audio.py`

Adds the voice-id validation relaxation, AudioBackend construction (with fail-fast), and the per-slide generate dispatch. Verifiable via the fail-fast error without a model.

- [ ] **Step 1: Relax voice-id validation for sherpa**

Around line 244, the validation is:
```python
    if args.provider != "edge" and not voice_id:
        parser.error(f"--voice-id is required for --provider {args.provider}")
        raise AssertionError("unreachable")
```
Change the condition to also exempt sherpa (sherpa defaults to speaker 0):
```python
    if args.provider not in ("edge", "sherpa") and not voice_id:
        parser.error(f"--voice-id is required for --provider {args.provider}")
        raise AssertionError("unreachable")
```

- [ ] **Step 2: Add the AudioBackend construction branch**

In the provider-selection chain (around line 279, after the `elif args.provider == "cosyvoice":` block and before the final `else:`), insert:
```python
    elif args.provider == "sherpa":
        try:
            backend_sherpa.read_sherpa_config()  # fail fast: model dir configured?
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        backend = AudioBackend(
            provider=args.provider,
            extension=backend_sherpa.output_extension(),
            voice_id=voice_id or "0",
        )
```
The surrounding `else:` (edge fallback) stays unchanged.

- [ ] **Step 3: Add the per-slide generate dispatch**

In the synthesis loop (around line 353-370), the existing chain ends with:
```python
            elif backend.provider == "cosyvoice":
                backend_cosyvoice.generate(
                    text,
                    output_path,
                    api_key=backend.api_key,
                    voice_id=backend.voice_id,
                    model=args.cosyvoice_model,
                    audio_format=args.cosyvoice_output_format,
                    sample_rate=args.cosyvoice_sample_rate,
                    volume=args.cosyvoice_volume,
                    rate=args.cosyvoice_rate,
                    pitch=args.cosyvoice_pitch,
                    instruction=args.cosyvoice_instruction,
                    language_hint=args.cosyvoice_language_hint,
                    base_url=args.cosyvoice_base_url,
                )
            else:
                asyncio.run(backend_edge.generate(text, output_path, voice=args.voice, rate=args.rate))
```
Insert a sherpa branch before `else:`:
```python
            elif backend.provider == "cosyvoice":
                backend_cosyvoice.generate(
                    text,
                    output_path,
                    api_key=backend.api_key,
                    voice_id=backend.voice_id,
                    model=args.cosyvoice_model,
                    audio_format=args.cosyvoice_output_format,
                    sample_rate=args.cosyvoice_sample_rate,
                    volume=args.cosyvoice_volume,
                    rate=args.cosyvoice_rate,
                    pitch=args.cosyvoice_pitch,
                    instruction=args.cosyvoice_instruction,
                    language_hint=args.cosyvoice_language_hint,
                    base_url=args.cosyvoice_base_url,
                )
            elif backend.provider == "sherpa":
                backend_sherpa.generate(
                    text,
                    output_path,
                    voice_id=backend.voice_id,
                    speed=args.sherpa_speed,
                )
            else:
                asyncio.run(backend_edge.generate(text, output_path, voice=args.voice, rate=args.rate))
```

- [ ] **Step 4: Verify fail-fast (no model needed)**

Create a throwaway project dir with one note, then run without `SHERPA_TTS_MODEL_DIR`:
```bash
cd /tmp && rm -rf sherpa-smoke && mkdir -p sherpa-smoke/notes
printf '你好，这是测试旁白。\n' > sherpa-smoke/notes/001.md
cd /home/sunziyun/ppt-master/skills/ppt-master/scripts
SHERPA_TTS_MODEL_DIR= python3 notes_to_audio.py /tmp/sherpa-smoke --provider sherpa
echo "exit=$?"
```
Expected: prints `error: Set SHERPA_TTS_MODEL_DIR=<sherpa-onnx 模型目录>...` to stderr and `exit=1`. This confirms fail-fast works before any model is present.

- [ ] **Step 5: Commit**
```bash
git add skills/ppt-master/scripts/notes_to_audio.py
git commit -m "feat(audio): wire sherpa runtime dispatch with fail-fast config check"
```

---

## Task 5: Document sherpa in `generate-audio.md`

**Files:**
- Modify: `skills/ppt-master/workflows/generate-audio.md`

Mirrors the file's existing English. Documents sherpa as the offline option and the manual model setup.

- [ ] **Step 1: Add dependency + setup note to *When to Run***

In `skills/ppt-master/workflows/generate-audio.md`, the *When to Run* list contains a `Default mode` bullet about edge-tts. After that block, add a new bullet:
```markdown
- Offline mode (`--provider sherpa`): install `sherpa-onnx` (`python3 -m pip install sherpa-onnx`) and manually place a Chinese VITS model in a directory, then set `SHERPA_TTS_MODEL_DIR` to that directory. The backend makes zero network calls; the model is not auto-downloaded.
```

- [ ] **Step 2: Add sherpa to *Step 2* backend guidance**

In Step 2, the opening sentence `Default to **edge** unless the user explicitly asks for a cloud provider...` — extend it:
```markdown
Default to **edge** unless the user explicitly asks for a cloud provider / higher-quality cloud narration / a cloned voice, or asks for **offline** synthesis (then use `--provider sherpa`, which needs `sherpa-onnx` installed and `SHERPA_TTS_MODEL_DIR` set).
```

- [ ] **Step 3: Add sherpa voice-catalog command to *Step 2***

After the existing cloud-provider `--list-voices` command block, add:
```markdown
**Offline sherpa backend**:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --provider sherpa --list-voices
```

sherpa voices are local speaker IDs (offline; no online catalog). Recommend speaker `0` for single-speaker Chinese VITS models.
```

- [ ] **Step 4: Add sherpa execution command to *Step 4***

In Step 4's command list (after the 1E CosyVoice block, before the `# 2.` re-export block), add a 1F block:
```markdown
# 1F. Or generate audio offline with sherpa-onnx (no network, no API key)
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider sherpa --voice 0 --sherpa-speed 1.0
```

- [ ] **Step 5: Commit**
```bash
git add skills/ppt-master/workflows/generate-audio.md
git commit -m "docs(audio): document sherpa offline provider in generate-audio"
```

---

## Task 6: Document sherpa in `native-enhance-pptx.md`

**Files:**
- Modify: `skills/ppt-master/workflows/native-enhance-pptx.md`

- [ ] **Step 1: Add sherpa to Step 7 voice confirmation**

In Step 7 (Audio Voice Confirmation), the prose says it follows the same standard as generate-audio. After the edge `--list-voices` example block, add:
```markdown
For offline sherpa voices:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --provider sherpa --list-voices
```
```

- [ ] **Step 2: Add sherpa to Step 8 generate command**

In Step 8, after the existing `notes_to_audio.py` example with `--voice`/`--rate`, add a sherpa variant:
```markdown
For the offline sherpa backend:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py "<project>" \
  --provider sherpa --voice 0 --sherpa-speed 1.0
```
```

- [ ] **Step 3: Commit**
```bash
git add skills/ppt-master/workflows/native-enhance-pptx.md
git commit -m "docs(audio): document sherpa offline provider in native-enhance-pptx"
```

---

## Task 7: End-to-end verification (manual, requires a model)

**Files:** none (verification only)

This task cannot run without a manually-prepared model, so it is a manual checklist, not a code change. Skip if no model is available; the backend is otherwise complete after Task 6.

- [ ] **Step 1: Obtain a Chinese VITS model**

Download a Chinese VITS model from the sherpa-onnx model catalog (GitHub `k2-fsa/sherpa-onnx` → TTS models → Chinese), e.g. a single-speaker `vits-zh-*` release. Extract it so the directory contains `model.onnx` (or `model.int8.onnx`) and `tokens.txt`.

- [ ] **Step 2: Generate audio for a smoke project**
```bash
export SHERPA_TTS_MODEL_DIR=/path/to/extracted/vits-zh
cd /home/sunziyun/ppt-master
python3 skills/ppt-master/scripts/notes_to_audio.py /tmp/sherpa-smoke \
  --provider sherpa --voice 0 --sherpa-speed 1.0
```
Expected: `[OK] /tmp/sherpa-smoke/audio/001.wav` then `[Done] Generated 1/1`.

- [ ] **Step 3: Confirm the WAV is valid and readable by ffprobe**
```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 /tmp/sherpa-smoke/audio/001.wav
```
Expected: a `duration=` line with a positive number (this is what `--recorded-narration audio` relies on).

- [ ] **Step 4 (optional): Confirm PPTX embed path**

If a full project exists with notes + exported deck:
```bash
python3 skills/ppt-master/scripts/svg_to_pptx.py <project_path> --recorded-narration audio
```
Expected: an `exports/<name>_<timestamp>_narrated.pptx` is produced (sherpa WAVs satisfy the same per-slide audio contract as other providers).

- [ ] **Step 5: No commit** — verification only.

---

## Verification summary

After Tasks 1-6, `--provider sherpa` is fully wired and fails fast without a model. Task 7 confirms real synthesis once a model is supplied. No automated tests are added (code-style §11).
