"""Local offline TTS backend (sherpa-onnx) for narration audio.

Pure-offline synthesis: the user prepares a sherpa-onnx VITS model once and
points SHERPA_TTS_MODEL_DIR at it. This module makes zero network calls.
"""

from __future__ import annotations

import array
import os
import wave
from pathlib import Path


def output_extension() -> str:
    """Audio extension produced by this backend (int16 WAV)."""
    return ".wav"


def read_sherpa_config() -> dict[str, Path]:
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


def _require_sherpa_onnx():
    """Import sherpa_onnx, soft-failing with install instructions (code-style §8)."""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `sherpa-onnx`. Install it with: "
            "python3 -m pip install sherpa-onnx"
        ) from exc
    return sherpa_onnx


def _build_config(model_dir: Path):
    """Build OfflineTtsConfig by probing the model dir for known files.

    Probes model.onnx / model.int8.onnx, tokens.txt, and optional lexicon.txt
    so the same code works across single/multi-speaker Chinese VITS models.
    """
    sherpa_onnx = _require_sherpa_onnx()

    model_file = model_dir / "model.onnx"
    if not model_file.exists():
        model_file = model_dir / "model.int8.onnx"
    if not model_file.exists():
        raise RuntimeError(
            f"未在 {model_dir} 找到 model.onnx 或 model.int8.onnx；"
            "请从 k2-fsa/sherpa-onnx 模型仓库下载中文 VITS 模型，"
            "解压到 SHERPA_TTS_MODEL_DIR 指向的目录"
        )

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise RuntimeError(
            f"未在 {model_dir} 找到 tokens.txt；请检查模型目录是否完整"
        )

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
        sherpa_onnx = _require_sherpa_onnx()
        cfg = read_sherpa_config()
        _get_tts._tts = sherpa_onnx.OfflineTts(_build_config(cfg["model_dir"]))
    return _get_tts._tts


def _write_wav(output_path: Path, samples, sample_rate: int) -> None:
    """Write float32 samples as a mono int16 WAV using only the stdlib."""
    data = array.array("h", (
        max(-32768, min(32767, int(float(sample) * 32767)))
        for sample in samples
    ))
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(data.tobytes())


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
    # voice_id 非数字时回退到 speaker 0：speaker 数量取决于具体模型，离线无法校验
    sid = int(voice_id) if str(voice_id).isdigit() else 0
    audio = tts.generate(text, sid=sid, speed=float(speed))
    _write_wav(output_path, audio.samples, audio.sample_rate)
