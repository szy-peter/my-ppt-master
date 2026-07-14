# 本地 TTS Backend 设计（sherpa-onnx）

- **日期**: 2026-07-14
- **状态**: 待实现（spec 已确认，待 writing-plans）
- **主题**: 为 ppt-master 新增第 6 个 TTS backend `sherpa`，提供纯离线旁白生成

## 1. 背景与目标

ppt-master 的旁白生成（[`generate-audio`](../../../skills/ppt-master/workflows/generate-audio.md) 与 [`native-enhance-pptx`](../../../skills/ppt-master/workflows/native-enhance-pptx.md) 工作流）目前依赖 5 个**联网** TTS 后端：

- `edge`（免费、无需 key，但每次合成走网络）
- `elevenlabs` / `minimax` / `qwen` / `cosyvoice`（需 API key + 网络）

全仓库没有任何 offline / 本地 TTS 的既有实现。本设计新增第 6 个 backend **`sherpa`**（基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)），提供**纯离线**旁白生成：首次手动准备模型后，合成过程零网络调用、零 API key。

**目标**：用户在断网或不愿调用云端 TTS 的场景下，仍能用现有 `notes_to_audio.py` 工作流为 PPT 生成中文旁白音频，且对现有 5 个 backend 零侵入。

## 2. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 目标定位 | 通用 backend | 可配置、跨机器、遵循 `code-style`、可回贡 upstream；不只个人自用 |
| 目标语言 | 中文为主 | 普通话 + 少量英文术语；选中文专用 VITS 模型，音质最佳、模型最小 |
| 硬件 | CPU 即可 | 锁定轻量引擎族，无 GPU / CUDA 依赖，跨平台 |
| 引擎 | sherpa-onnx | 纯 CPU、中文 VITS-zh 质量好、`pip` 一行安装、模型小、接入接口干净 |
| provider 命名 | `sherpa` | 与现有 5 个按引擎/服务命名风格一致；避免未来第二个本地引擎命名冲突（`local`/`offline` 会冲突） |
| 模型获取 | **不自动下载** | 代码零网络调用，名实相符；用户手动准备模型，可控可复现 |

## 3. 架构与数据流

新增第 6 个 backend，完全遵循现有"插件式分发"模式——不引入注册表、不改既有 provider 行为、零侵入。

```
notes/*.md ──spoken_text()──► text
   │  for each slide, provider == "sherpa"
   ▼
backend_sherpa.generate(text, output_path, voice_id=…, speed=…)
   │  sherpa_onnx.OfflineTts   ← 懒加载 + 进程内单例（多页只加载一次模型）
   ▼
audio/<stem>.wav   (001.wav, 002.wav, …)   ← 文件名与 note 同名
```

**关键设计原则**：
1. 模型懒加载并缓存到函数属性——多页循环只载一次，否则每页重载会慢到不可用。
2. 输出 `.wav`——`backend_common` 只认 mp3/wav，本地合成天然产 wav，最省事。
3. 沿用现有 `--voice` / `--voice-id` 语义，不新造交互参数（除引擎专属的 `--sherpa-speed`）。

## 4. `backend_sherpa.py` 模块接口

文件路径：`skills/ppt-master/scripts/tts_backends/backend_sherpa.py`

导出函数契约（对照现有 backend 模式；本地无 API key，用"读模型路径配置"替代 `read_*_api_key`）：

| 函数 | 签名 | 作用 |
|---|---|---|
| `output_extension()` | `-> ".wav"` | 固定 wav，符合 `backend_common` 格式约束 |
| `read_sherpa_config()` | `-> dict` | 从 `SHERPA_TTS_MODEL_DIR` 读路径，缺失则抛清晰错误 |
| `_get_tts()` | 内部 | 懒加载 `OfflineTts` 并缓存到函数属性 |
| `generate(text, output_path, *, voice_id, speed, **_)` | 核心合成 | 调 `_get_tts().generate()`，用标准库 `wave` 写 int16 wav |
| `print_voices()` | 打印清单 | 离线无在线清单，打印 speaker id 说明 |

**配置方式**（单一目录约定，最简最通用）：
- 一个环境变量 `SHERPA_TTS_MODEL_DIR` 指向模型目录
- 目录内按约定放：`model.onnx`、`tokens.txt`、`lexicon.txt`（中文 G2P，可选）
- backend 在该目录按约定文件名查找，缺失文件给清晰错误提示
- **不做自动下载**——代码里无任何网络请求；用户手动准备模型

**参数映射**（与现有多参数 backend 一致，各自专属参数）：
- `--voice <id>`：speaker id（数字），默认 `"0"`。单说话人模型只用 0；多说话人模型（如 aishell3）可传 0~N
- `--sherpa-speed <float>`：语速倍率，默认 `1.0`（不复用 edge 的 `--rate`，因格式不兼容；和 `--cosyvoice-rate` / `--minimax-speed` 模式一致）

**骨架**（聚焦契约；`OfflineTtsConfig` 的确切字段以所选 sherpa-onnx 版本和模型为准，实现期验证）：

```python
"""Local offline TTS backend (sherpa-onnx) for narration audio."""
from __future__ import annotations

import os
import struct
import wave
from pathlib import Path


def output_extension() -> str:
    return ".wav"


def read_sherpa_config() -> dict:
    model_dir = os.environ.get("SHERPA_TTS_MODEL_DIR", "").strip()
    if not model_dir:
        raise RuntimeError("Set SHERPA_TTS_MODEL_DIR=<sherpa-onnx 模型目录>")
    return {"model_dir": Path(model_dir)}


def _get_tts():
    """懒加载 + 进程内单例（多页循环只载一次模型）。"""
    if not hasattr(_get_tts, "_tts"):
        import sherpa_onnx
        cfg = read_sherpa_config()
        _get_tts._tts = sherpa_onnx.OfflineTts(_build_config(cfg["model_dir"]))
    return _get_tts._tts


def generate(text, output_path: Path, *, voice_id: str = "0", speed: float = 1.0, **_):
    tts = _get_tts()
    sid = int(voice_id) if str(voice_id).isdigit() else 0
    audio = tts.generate(text, sid=sid, speed=float(speed))
    _write_wav(output_path, audio.samples, audio.sample_rate)


def print_voices() -> None:
    print("sherpa-onnx speakers (本地模型 speaker id):")
    print("ID   Note")
    print("0    默认说话人（单说话人模型唯一可选）")
    print("0..N 多说话人模型(如 aishell3)可选 0~N，取决于模型")
```

> `_build_config` / `_write_wav` 为模块内部辅助：`_build_config` 构造 `OfflineTtsConfig`（字段如 `vits`/`lexicon`/`data_dir`/`num_threads`/`provider="cpu"` 取决于具体中文模型），`_write_wav` 用标准库 `wave` 把 float32 samples 转 int16 写盘。两者均为实现期验证项，不影响对外契约。

## 5. `notes_to_audio.py` 集成（7 处改动）

均为"照葫芦画瓢"新增分支，零侵入。行号为参考，实现时按当前文件对齐。

**① import（约 line 35）**
```python
from tts_backends import (
    backend_cosyvoice,
    backend_edge,
    backend_elevenlabs,
    backend_minimax,
    backend_qwen,
    backend_sherpa,   # 新增
)
```

**② `_load_tts_env_file` 加前缀（约 line 54）**——让 `.env` 里的 `SHERPA_TTS_MODEL_DIR` 生效：
```python
load_prefixed_env_file((
    "ELEVENLABS_", "MINIMAX_", "QWEN_", "DASHSCOPE_", "COSYVOICE_",
    "SHERPA_TTS_",   # 新增
))
```

**③ 新增 `--sherpa-speed` 参数（约 line 202 之后）**
```python
parser.add_argument("--sherpa-speed", type=float, default=1.0,
                    help="sherpa-onnx 语速倍率（默认 1.0）")
```

**④ `--list-voices` 分发（约 line 222）**
```python
elif args.provider == "sherpa":
    backend_sherpa.print_voices()
```

**⑤ `voice_id` 校验放宽（约 line 244）**——sherpa 不强制 `--voice-id`，默认 speaker 0：
```python
# 原: if args.provider != "edge" and not voice_id:
if args.provider not in ("edge", "sherpa") and not voice_id:
    parser.error(f"--voice-id is required for --provider {args.provider}")
```

**⑥ 构造 `AudioBackend`（约 line 279）**——fail fast：生成前先校验模型目录存在：
```python
elif args.provider == "sherpa":
    try:
        backend_sherpa.read_sherpa_config()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    backend = AudioBackend(
        provider=args.provider,
        extension=backend_sherpa.output_extension(),
        voice_id=voice_id or "0",
    )
```

**⑦ 主循环 `generate` 调用（约 line 368）**
```python
elif backend.provider == "sherpa":
    backend_sherpa.generate(
        text,
        output_path,
        voice_id=backend.voice_id,
        speed=args.sherpa_speed,
    )
```

`--provider` 无 `choices=` 限制（已确认），所以不用改参数定义。

## 6. 错误处理边界

| 失败场景 | 处理 | 对齐 |
|---|---|---|
| `sherpa-onnx` 未安装 | `generate` 内 `import` 抛 `RuntimeError` | 仿 edge：提示 `pip install sherpa-onnx` |
| `SHERPA_TTS_MODEL_DIR` 未设 | `read_sherpa_config()` 抛清晰错误 | fail fast，改动⑥生成前拦截 |
| 目录缺 `model.onnx` / `tokens.txt` | `_get_tts()` 加载抛错，透传 sherpa 信息 | 提示检查目录文件 |
| speaker id 越界（多说话人模型） | 透传 sherpa 错误 | — |
| 空文本 | 主循环已有 `spoken_text` 检查 | skip 该页 |

**fail fast 原则**：模型目录没配时，在生成第一页前就报错退出（改动⑥），而不是跑了几页才崩——与现有 backend 报 API key 缺失的位置一致。

## 7. 首次使用引导（手动）

写进 `generate-audio.md` 的 *When to Run*。**主语是用户（手动操作），代码不做任何下载**：

1. `python3 -m pip install sherpa-onnx`
2. **手动**从 sherpa-onnx 模型仓库下载一个中文 VITS 模型，解压到本地目录
3. `export SHERPA_TTS_MODEL_DIR=<解压目录>`（或写进 `.env`）
4. `python3 skills/ppt-master/scripts/notes_to_audio.py --provider sherpa --list-voices` 验证配置

## 8. workflow 文档同步

新增内容镜像目标文件语言（英文），遵循 AGENTS.md markdown 语言一致性。

**`skills/ppt-master/workflows/generate-audio.md`**：
- *When to Run*：加 sherpa-onnx 依赖说明 + `SHERPA_TTS_MODEL_DIR`，说明模型需手动准备
- *Step 2*（后端选择）：把 sherpa 列为**离线**选项
- *Step 3*（交互模板）：加 sherpa 候选
- *Step 4*（执行命令）：加 `--provider sherpa` 示例（1F）

**`skills/ppt-master/workflows/native-enhance-pptx.md`**：
- *Step 7*（音色确认）：backend 选项加 sherpa
- *Step 8*（生成命令）：加 sherpa 示例

**关键定位语**（文档里写清，帮 AI 在确认阶段选择）：
- `sherpa` = **离线**（首次手动准备模型后纯本地，无网络、无 key）
- `edge` = **联网免费**（无 key 但每次合成走网络）

## 9. 验证方案

遵循 [`docs/rules/code-style.md`](../../rules/code-style.md) §11——本 repo **不带自动化测试**（无 `tests/`、无 `test_*.py`、无 pytest/unittest）。验证纯手动：

- **无模型契约验证**（不需真实模型）：
  - `--provider sherpa --list-voices` 验证 `print_voices` 输出
  - 不设 `SHERPA_TTS_MODEL_DIR` 跑一次，确认报错文案清晰正确
- **端到端验证**（需手动准备的模型）：
  - 准备一个中文 VITS 模型
  - `python3 skills/ppt-master/scripts/notes_to_audio.py <project> --provider sherpa`
  - 检查 `audio/*.wav` 生成、`ffprobe` 能读时长
  - 可选：`python3 skills/ppt-master/scripts/svg_to_pptx.py <project> --recorded-narration audio` 验证嵌入导出

## 10. code-style 遵循

- `backend_sherpa.py`：模块 docstring 英文、`from __future__ import annotations`、函数顺序对齐 `backend_qwen.py`；面向用户的语音描述可中文（与 `backend_edge.py` 的 `COMMON_VOICES` 一致）
- `notes_to_audio.py`：保持 `main(argv=None) -> int` 可测约定、沿用已有 `configure_utf8_stdio()`，UTF-8 stdio 不变
- 不写自动化测试（§11 硬规则）

## 11. 非目标（YAGNI）

明确**不做**以下事项，避免范围蔓延：

- 不做模型自动下载（代码零网络调用）
- 不做多引擎切换（单 `sherpa` backend；未来第二个本地引擎另起 backend，如 `melotts`）
- 不做自动化测试（repo 约定）
- 不做 GUI / 配置向导（沿用 CLI + 环境变量）
- 不支持 GPU 推理路径（CPU 即可；如需 GPU 走 sherpa-onnx 自身的 provider 配置，不在本 backend 内特殊处理）
