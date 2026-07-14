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
