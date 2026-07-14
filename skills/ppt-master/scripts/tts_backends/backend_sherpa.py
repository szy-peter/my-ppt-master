"""Remote sherpa-onnx TTS backend for narration audio.

Calls an in-network HTTP server running ``sherpa_server.py`` (which loads the
VITS model). This client module performs no local sherpa-onnx inference and
needs no model on this machine — only ``SHERPA_TTS_SERVER_URL`` pointing at the
server. Communication stays inside the intranet: no external network, no API
key.
"""

from __future__ import annotations

import os
from pathlib import Path

from tts_backends.backend_common import get_json, post_and_download


def output_extension() -> str:
    """Audio extension produced by this backend (int16 WAV)."""
    return ".wav"


def read_sherpa_config() -> dict[str, str]:
    """Read the inference-server URL from the environment. Fail fast if unset."""
    url = os.environ.get("SHERPA_TTS_SERVER_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError(
            "Set SHERPA_TTS_SERVER_URL=http://<内网ip>:<port>"
            "（指向部署了 sherpa_server.py 的推理服务器）"
        )
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(
            f"SHERPA_TTS_SERVER_URL 必须以 http:// 或 https:// 开头: {url}"
        )
    return {"server_url": url}


def print_voices() -> None:
    """Print speaker info. Queries the server if reachable; else static fallback."""
    try:
        cfg = read_sherpa_config()
        info = get_json(f"{cfg['server_url']}/voices", timeout=10)
        model = info.get("model_dir", "?")
        arch = info.get("architecture", "?")
        n = info.get("num_speakers")
        sr = info.get("sample_rate")
        upper = (n - 1) if isinstance(n, int) else "N-1"
        print(f"sherpa server model: {model} ({arch})")
        print(f"speakers: {n} @ {sr} Hz, speaker id range 0..{upper}")
        hint = _voice_hint(model)
        if hint:
            print(hint)
    except Exception:
        print("sherpa-onnx speakers (未能连接服务器查询，使用静态说明):")
        print("ID   Note")
        print("0    默认说话人（单说话人模型唯一可选）")
        print("0..N 多说话人模型可选 0~N-1，取决于服务端模型")
        print("（确认 SHERPA_TTS_SERVER_URL 已配且 sherpa_server.py 正在运行）")


def _voice_hint(model_dir: str) -> str:
    """Return a male/female sid-range hint for known kokoro models."""
    name = (model_dir or "").lower()
    if "kokoro-multi-lang-v1_1" in name:
        return "中文女声 zf: sid 3-57；中文男声 zm: sid 58-102"
    if "kokoro-multi-lang-v1_0" in name:
        return "中文女声 zf / 中文男声 zm: sid 0-52（详见 sherpa-onnx 文档）"
    return ""


def generate(
    text: str,
    output_path: Path,
    *,
    voice_id: str = "0",
    speed: float = 1.0,
    **_,
) -> None:
    """Synthesize one note via the remote sherpa server into a WAV file."""
    cfg = read_sherpa_config()
    # voice_id 非数字时回退到 speaker 0：speaker 数量取决于服务端模型，客户端无法校验
    sid = int(voice_id) if str(voice_id).isdigit() else 0
    wav_bytes = post_and_download(
        f"{cfg['server_url']}/tts",
        payload={"text": text, "sid": sid, "speed": float(speed)},
    )
    Path(output_path).write_bytes(wav_bytes)
