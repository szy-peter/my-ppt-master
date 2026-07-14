#!/usr/bin/env python3
"""sherpa-onnx offline TTS HTTP server.

Deploy on the inference machine (the one holding the VITS model). The ppt-master
client (``backend_sherpa.py``) calls this server over the intranet via
``SHERPA_TTS_SERVER_URL``.

Run::

    SHERPA_TTS_MODEL_DIR=/path/to/vits-zh \\
    uvicorn sherpa_server:app --host 0.0.0.0 --port 8300

Dependencies (server machine only)::

    python3 -m pip install sherpa-onnx fastapi uvicorn pydantic

The model is loaded once at startup (FastAPI lifespan) so configuration errors
fail fast and the first request is fast.

Security: bind only to a trusted intranet — the port is unauthenticated.
"""

from __future__ import annotations

import array
import os
import wave
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


def _resolve_model_dir() -> Path:
    """Read SHERPA_TTS_MODEL_DIR. Fail fast if unset/not a directory."""
    model_dir = os.environ.get("SHERPA_TTS_MODEL_DIR", "").strip()
    if not model_dir:
        raise RuntimeError(
            "Set SHERPA_TTS_MODEL_DIR=<sherpa-onnx 模型目录>"
            "（手动下载模型后解压到此目录，代码不做自动下载）"
        )
    path = Path(model_dir)
    if not path.is_dir():
        raise RuntimeError(f"SHERPA_TTS_MODEL_DIR 不是目录: {model_dir}")
    return path


def _find_model_file(model_dir: Path) -> Path:
    """Find the model .onnx (prefer model.onnx / model.int8.onnx, else any *.onnx)."""
    for candidate in (model_dir / "model.onnx", model_dir / "model.int8.onnx"):
        if candidate.exists():
            return candidate
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(
            f"未在 {model_dir} 找到任何 .onnx 模型文件；"
            "请从 k2-fsa/sherpa-onnx 模型仓库下载模型，"
            "解压到 SHERPA_TTS_MODEL_DIR 指向的目录"
        )
    non_int8 = [p for p in onnx_files if ".int8." not in p.name]
    return (non_int8 or onnx_files)[0]


def _collect_rule_fsts(model_dir: Path) -> str:
    """Collect text-normalization .fst files (kokoro uses *-zh.fst, vits uses *.fst)."""
    names = ("number-zh.fst", "phone-zh.fst", "date-zh.fst",
             "number.fst", "phone.fst", "date.fst")
    return ",".join(str(model_dir / n) for n in names if (model_dir / n).exists())


def _build_config(model_dir: Path):
    """Build OfflineTtsConfig, auto-detecting Kokoro vs VITS architecture.

    Kokoro models (e.g. kokoro-multi-lang-v1_1) carry voices.bin + espeak-ng-data
    and use OfflineTtsKokoroModelConfig. VITS models (e.g. vits-zh-hf-*) carry
    lexicon.txt + dict/ and use OfflineTtsVitsModelConfig. Detection key:
    presence of voices.bin.
    """
    import sherpa_onnx

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise RuntimeError(f"未在 {model_dir} 找到 tokens.txt；请检查模型目录是否完整")

    model_file = _find_model_file(model_dir)
    rule_fsts = _collect_rule_fsts(model_dir)
    dict_dir_str = str(model_dir / "dict") if (model_dir / "dict").is_dir() else ""

    if (model_dir / "voices.bin").exists():
        # Kokoro architecture (voices + espeak-ng-data multi-lang G2P)
        espeak = model_dir / "espeak-ng-data"
        lexicons = sorted(model_dir.glob("lexicon-*.txt"))
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(model_file),
            voices=str(model_dir / "voices.bin"),
            tokens=str(tokens),
            data_dir=str(espeak) if espeak.is_dir() else "",
            lexicon=",".join(str(p) for p in lexicons),
            dict_dir=dict_dir_str,
        )
        tts_model = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=kokoro, num_threads=1, debug=False, provider="cpu",
        )
    else:
        # VITS architecture (lexicon + dict jieba)
        lexicon = model_dir / "lexicon.txt"
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model_file),
            tokens=str(tokens),
            lexicon=str(lexicon) if lexicon.exists() else "",
            dict_dir=dict_dir_str,
        )
        tts_model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits, num_threads=1, debug=False, provider="cpu",
        )

    return sherpa_onnx.OfflineTtsConfig(
        model=tts_model,
        max_num_sentences=2,
        rule_fsts=rule_fsts,
    )


def _encode_wav(samples, sample_rate: int) -> bytes:
    """Encode float32 samples as mono int16 WAV bytes."""
    data = array.array("h", (
        max(-32768, min(32767, int(float(sample) * 32767))) for sample in samples
    ))
    buf = BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(data.tobytes())
    return buf.getvalue()


_state: dict = {"tts": None}


def _get_tts():
    """Return the cached OfflineTts singleton (loaded at startup via lifespan)."""
    if _state["tts"] is None:
        import sherpa_onnx
        _state["tts"] = sherpa_onnx.OfflineTts(_build_config(_resolve_model_dir()))
    return _state["tts"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model once at startup so config errors fail fast.
    _get_tts()
    yield


app = FastAPI(title="sherpa-onnx TTS", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    sid: int = Field(0, ge=0)
    speed: float = Field(1.0, gt=0, le=4.0)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/voices")
def voices() -> dict:
    """Return the loaded model's speaker info so clients can pick a voice id."""
    model_dir = _resolve_model_dir()
    tts = _get_tts()
    return {
        "num_speakers": tts.num_speakers,
        "sample_rate": tts.sample_rate,
        "model_dir": model_dir.name,
        "architecture": "kokoro" if (model_dir / "voices.bin").exists() else "vits",
    }


@app.post("/tts")
def synthesize(req: TtsRequest) -> Response:
    try:
        audio = _get_tts().generate(req.text, sid=req.sid, speed=req.speed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=_encode_wav(audio.samples, audio.sample_rate),
        media_type="audio/wav",
    )
