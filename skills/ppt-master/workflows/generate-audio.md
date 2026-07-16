---
description: Generate per-slide narration audio with AI-recommended voice selection, then optionally re-export PPTX with embedded audio
---

# Generate Audio Workflow

> Standalone post-export step. Run when the user asks for "生成音频" / "录制旁白" / "narrated PPT" / "video export with voice", or proactively offer it after a deck is exported. Produces one audio file per slide via the offline `sherpa` backend by default (intranet; needs `SHERPA_TTS_SERVER_URL`), or `edge-tts` / a cloud TTS provider (`elevenlabs` / `minimax` / `qwen` / `cosyvoice`) when the user chooses online or higher-quality narration or a cloned voice, then optionally re-exports a video-ready PPTX with audio embedded and per-slide auto-advance timings.

This workflow is **independent**: it reads `notes/*.md` and queries the selected TTS voice catalog — no upstream conversation context required. Safe to invoke in a fresh session.

## When to Run

- `notes/total.md` exists and has been split into per-page files at `notes/*.md` (post-processing Step 7.1 done).
- Default mode (offline `sherpa`): run the `sherpa_server.py` FastAPI service on an inference machine that holds a Chinese VITS model, then point the client at it via `SHERPA_TTS_SERVER_URL`. Server setup: `SHERPA_TTS_MODEL_DIR=<vits-zh dir> uvicorn sherpa_server:app --host 0.0.0.0 --port 8300` (server deps: `sherpa-onnx fastapi uvicorn pydantic`). The client needs only `SHERPA_TTS_SERVER_URL`; no local model, no external network, no API key.
- Online mode (`--provider edge`): `edge-tts` is installed (`python3 -m pip install edge-tts`).
- The workflow is page-level only: one notes file becomes one audio file. Do not use a single long audio track or attempt automatic long-audio splitting.
- PPT narration assets must be PowerPoint-reliable audio: `m4a` (AAC), `mp3`, or `wav`. The built-in TTS path defaults to `mp3`; provider formats such as `pcm`, `opus`, or `flac` must be transcoded before embedding.
- PowerPoint recorded narration export requires `ffprobe` so slide timings can be written from actual audio duration.
- High-quality cloud mode: provider API key is set before use:
  - ElevenLabs: `ELEVENLABS_API_KEY`
  - MiniMax: `MINIMAX_API_KEY`
  - Qwen: `QWEN_API_KEY` or `DASHSCOPE_API_KEY`
  - CosyVoice: `COSYVOICE_API_KEY` or `DASHSCOPE_API_KEY`
  - Keys may live in the current process environment or the first `.env` found in this order: current working directory, skill directory (e.g. `~/.agents/skills/ppt-master/.env`), clone repo root, `~/.ppt-master/.env`
- The deck is in a single dominant language (mixed-language decks: pick the dominant one — the AI uses judgment, not a heuristic).

If `notes/*.md` are missing, run `total_md_split.py <project_path>` first.

---

## Step 1: Determine the deck's language

The AI already knows the deck's language from writing the notes. No detection script needed.

- Identify the primary language from the notes content: `zh` / `en` / `ja` / `ko` / etc.
- For mixed-language decks (e.g. Chinese with English technical terms), pick the language the audience will hear most of.
- For Chinese specifically: pick the locale based on context — `zh-CN` (mainland mandarin, default), `zh-TW` (Taiwanese mandarin), or `zh-HK` (Cantonese). Ask the user only if the project context doesn't make it clear.

---

## Step 2: Probe environment, then pull available voice catalogs

The backends offered are **connectivity-aware**: online backends (`edge` + cloud providers) appear only when the environment has internet, and each cloud provider appears only when its API key is configured. Probe first, then list voices only for backends the probe found available.

**Probe** (run once; the `.env` search order is listed in *When to Run* above):

```bash
# 1. Internet connectivity — decides the online/offline branch
timeout 5 curl -fsS -o /dev/null https://www.bing.com && echo ONLINE || echo OFFLINE

# 2. edge-tts installed?
python3 -c "import edge_tts" 2>/dev/null && echo "edge-tts: installed" || echo "edge-tts: missing"

# 3. Cloud provider keys configured (checks each .env in the search order)
for f in ./.env skills/ppt-master/.env ~/.ppt-master/.env; do
  [ -f "$f" ] || continue
  for k in ELEVENLABS_API_KEY MINIMAX_API_KEY QWEN_API_KEY DASHSCOPE_API_KEY COSYVOICE_API_KEY; do
    grep -qE "^$k=.+" "$f" && echo "$k: configured ($f)"
  done
done

# 4. sherpa intranet server reachable?
python3 skills/ppt-master/scripts/notes_to_audio.py --provider sherpa --list-voices 2>&1 | head -3
```

**Menu assembly** — build the offered-backends set from the probe:

| Probe result | Backends offered |
|---|---|
| ONLINE + edge-tts installed + ≥1 cloud key | sherpa (if server ok) · edge · each cloud provider with a key |
| ONLINE + edge-tts installed, no cloud keys | sherpa (if server ok) · edge — name the cloud providers that still need a key |
| ONLINE + edge-tts missing | sherpa (if server ok) · each cloud provider with a key — note edge-tts not installed |
| OFFLINE | sherpa only (if server ok) — `edge` and all cloud providers need public internet, exclude them |

**Default — sherpa (may override)**: recommend `sherpa` (offline/intranet, no API key, no external network) when its server is reachable, even when online. But the *menu* must include every online backend the probe found — do not hide a viable online option behind the default.

**Pull voice catalogs** only for backends the probe found available:

**edge backend**:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --list-voices --locale <locale>
```

**ElevenLabs backend**:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --provider elevenlabs --list-voices
```

**Cloud providers using explicit voice IDs/names**:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --provider minimax --list-voices
python3 skills/ppt-master/scripts/notes_to_audio.py --provider qwen --list-voices
python3 skills/ppt-master/scripts/notes_to_audio.py --provider cosyvoice --list-voices
```

**Intranet sherpa backend**:

```bash
python3 skills/ppt-master/scripts/notes_to_audio.py --provider sherpa --list-voices
```

sherpa voices are speaker IDs of the server's model (no online catalog). For the current **kokoro-multi-lang** model, Chinese voices are sid 3–102 (female 3–57, male 58–102); the repo default is **sid 58 (中文男声)**.

The output is a flat list of all available voices for the selected provider. From this list, the AI picks **3–6 candidates** to recommend, applying these rules:

- **Cover both genders** when both exist for the locale.
- **For edge**: prefer `COMMON_VOICES`-listed voices (curated set inside `notes_to_audio.py`) when the locale has them — they are battle-tested.
- **For ElevenLabs**: prefer voices already present in the user's account; if the user provides a specific `voice_id`, do not override it.
- **For MiniMax / Qwen / CosyVoice**: if the user provides a cloned `voice_id`, use it directly. Do not attempt voice cloning inside the narration workflow.
- **Match the deck's tone** — pick the strongest recommendation based on style:
  - Consultant / data-driven / 财报 → 稳重男声（如 `zh-CN-YunjianNeural`）or 清晰女声（如 `zh-CN-XiaoxiaoNeural`）
  - General / 教学 / 产品介绍 → 明亮女声 / 年轻男声（如 `zh-CN-XiaoyiNeural` / `zh-CN-YunxiNeural`）
  - 发布会 / 播报 → 播报感男声（如 `zh-CN-YunyangNeural`）
  - English consultant deck → `en-US-GuyNeural` (steady) or `en-US-JennyNeural` (clear)
  - Japanese / Korean → pick from `ja-JP-*` / `ko-KR-*` neural voices, mark gender + tone

For each candidate, write a **one-line Chinese description** covering: 性别 · 调性 · 适用场景。For cloud providers, include the voice name/ID exactly as it must be passed to `--voice-id`.

---

## Step 3: One-shot user interaction (mandatory)

Send a single message to the user that asks all three questions at once and provides a recommended value for each. Do NOT split into multiple rounds.

**Cloned-voice fast path**: if the user mentioned a cloned voice / 克隆音色 / 复刻音色 / "my own voice" along with a `voice_id`, skip the voice-recommendation list — set the provider to whichever the user named (`elevenlabs` / `minimax` / `qwen` / `cosyvoice`), pin the `voice_id` they gave you, and only confirm rate + embed-or-not.

**Offered options are probe-bound**: the 生成模式 candidates and the voice list come from the Step 2 probe. OFFLINE → the menu is sherpa-only (do not offer `edge` or cloud providers). ONLINE → include every backend the probe found available, and name any cloud provider that needs a key the user has not yet configured.

**Message template** (Chinese; translate to user's chat language if different):

> 检测到 notes 主语言为 **<语言>**（locale: `<locale>`）。基于 deck 调性（<风格>），我推荐以下配置：
>
> **生成模式**：⭐ 推荐 `<sherpa|edge|elevenlabs|minimax|qwen|cosyvoice>`（理由：<一句话，如"离线稳定无需 Key"或"用户要在线/高质量云端音色">）。
>
> **音色**：
> - **[1] <ShortName>** — <性别·调性·适用场景> ⭐ **推荐**
> - [2] <ShortName> — <性别·调性·适用场景>
> - [3] <ShortName> — <性别·调性·适用场景>
> - [4] <ShortName> — <性别·调性·适用场景>
> - [5] <ShortName> — <性别·调性·适用场景>
> - 也可直接输入清单中的其他 ShortName。
>
> **语速/风格参数**：⭐ 推荐 `<rate or provider defaults>`（理由：<一句话，如"页均 2–3 句，正常语速听感最稳"或"ElevenLabs 默认 voice settings 保留音色原始表现最稳">）。
>
> **生成完是否重新导出嵌入音频的 PPTX**：⭐ 推荐 **是**（一次到位，自动按音频时长设页面停留）。
>
> 直接回"好"用全部推荐值，或告诉我想改的部分（如"音色 2，语速 -5%"或"用 MiniMax 的 voice_id xxx"）。

**Recommended-value rules**:
- 生成模式：默认推荐 `sherpa`（离线/内网，无需 API Key、不连外网）。Step 2 探测为 ONLINE 时，菜单须同时列出 `edge` 与已配 Key 的云端后端供用户选择；用户选了在线后端，按其指定用 `edge` / `elevenlabs` / `minimax` / `qwen` / `cosyvoice`。OFFLINE 时只给 `sherpa`。
- 音色：从 Step 2 候选里挑最贴合 deck 调性的那一个。
- 语速：sherpa 用 `--sherpa-speed 1.0`（倍率，1.0 为正常）；edge 默认 `+0%`，notes 字数密集（页均 >4 句长句）建议 `-5%`，简短紧凑建议 `+5%`，超出此范围需说明理由。Cloud providers 默认用 provider defaults，除非用户明确要调速或改风格。
- 嵌入：默认推荐"是"；除非用户已有定制 PPTX 不希望覆盖。

---

## Step 4: Execute (no further interaction)

Run sequentially — do NOT bundle:

> **Run in the background by default** (`run_in_background: true`) — serial sherpa TTS can exceed the 10-min foreground cap.

```bash
# 1A. Generate audio with sherpa (default, offline/intranet; no API key, no external network)
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider sherpa --voice 0 --sherpa-speed 1.0

# 1B. Or generate audio with ElevenLabs
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider elevenlabs --voice-id <chosen-voice-id> \
  --elevenlabs-model eleven_multilingual_v2

# 1C. Or generate audio with MiniMax
# Defaults to the China endpoint; set MINIMAX_TTS_BASE_URL=https://api.minimax.io/v1/t2a_v2 for overseas access.
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider minimax --voice-id <chosen-voice-id> \
  --minimax-model speech-2.8-hd

# 1D. Or generate audio with Qwen TTS
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider qwen --voice-id <chosen-voice> \
  --qwen-model qwen3-tts-flash --qwen-language-type Chinese

# 1E. Or generate audio with CosyVoice
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider cosyvoice --voice-id <chosen-voice> \
  --cosyvoice-model cosyvoice-v3-flash

# 1F. Or generate audio with edge (online, no API key)
python3 skills/ppt-master/scripts/notes_to_audio.py <project_path> \
  --provider edge --voice <chosen-ShortName> --rate <chosen-rate>

# 2. (If user kept embedding) Re-export PPTX with audio embedded
python3 skills/ppt-master/scripts/svg_to_pptx.py <project_path> \
  --recorded-narration audio
```

If `notes_to_audio.py` errors with a missing dependency or missing provider API key, fix the prerequisite and re-run — do NOT swallow the error.

`--recorded-narration audio` prepares PowerPoint's recorded timings and narrations: every slide must have a matching supported audio file, every duration must be readable by `ffprobe`, and object animations must not use `--animation-trigger on-click`. Use `after-previous` or `with-previous` for narrated/video export. Narration changes the slide-advance layer only: the resolved page-transition effect remains unchanged, `-t none` remains visually transition-free, and narration advance disables click while using audio duration plus padding. The re-export is saved as `exports/<project_name>_<timestamp>_narrated.pptx`, telling it apart from silent exports.

---

## Step 5: Completion report

Output one summary block listing:

- Number of audio files generated and their location (`<project_path>/audio/*`).
- The provider, voice, and rate/settings actually used.
- (If embedded) the new narrated PPTX path under `<project_path>/exports/`.
- (If skipped embedding) one-line hint on how to embed later: `python3 skills/ppt-master/scripts/svg_to_pptx.py <project_path> --recorded-narration audio`.
