# Kurage Montage

Kurage Montage (`kmontage`) turns a reference X, YouTube, blog/article, or PDF URL into a Japanese short explainer video through Kurage.

## Structure

- `backend/`: FastAPI app for reference video analysis and Kurage job orchestration.
- `static/`: single-page UI.
- `storage/jobs/`: local job metadata. Downloaded media is ignored.
- `OpenMontage/`: upstream reference OSS clone, intentionally ignored by Git.

## Flow

1. User inputs an X, YouTube, blog/article, or PDF URL.
2. Video URLs use `yt-dlp`, fxtwitter, and Kurage Voice Pro transcription when needed. Article/blog/PDF URLs extract source text directly.
3. Source text, captions, or Whisper transcription are analyzed by Ollama into OpenMontage-style artifacts:
   `reference_analysis.json`, `scene_plan.json`, `script.json`, and `qa.json`.
4. The completed script is sent to Kurage `/generate_from_script` with VTuber mode enabled.

Before enqueueing to Kurage, kmontage now applies a Japanese quality gate. If
the generated title or narration remains English, or if the script has too few
scenes, the backend asks Ollama to rebuild the analysis as a Japanese 12-scene
short script. If the repaired script still fails the Japanese check, the job is
stopped instead of publishing an English video.
5. The resulting video appears in `kuragev.php` because Kurage owns the final job JSON/video.

`/generate_from_news` is intentionally not used for kmontage reference-video jobs.
It can rewrite the source into a generic news explainer and dilute concrete
numbers, tools, workflow steps, and risks from the original video. kmontage must
produce the faithful script first, then ask Kurage only to render it.

## Run

```bash
cd /home/kojima/work/kmontage
/home/kojima/work/kuragevp/.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 18305
```

Open:

```text
http://localhost:18305/
```

## Environment

```bash
KURAGE_API=http://127.0.0.1:18303
OLLAMA_URL=http://192.168.0.3:11434
OLLAMA_MODEL=gemma4:12b-it-qat
KURAGEVP_BACKEND_DIR=/home/kojima/work/kuragevp/backend
KURAGEVP_WHISPER_DEVICE=cuda
KURAGEVP_WHISPER_COMPUTE_TYPE=float16
LD_LIBRARY_PATH=/home/kojima/work/kuragevp/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:/home/kojima/work/kuragevp/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/kojima/work/kuragevp/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
KMONTAGE_YTDLP_COOKIES_BROWSER=chrome
# or
KMONTAGE_YTDLP_COOKIES_FILE=/path/to/cookies.txt
```

X動画は未ログイン状態だと取得できないことがあります。その場合は、認証済みブラウザを使っている環境で
`KMONTAGE_YTDLP_COOKIES_BROWSER=chrome` か、書き出したCookieファイルを
`KMONTAGE_YTDLP_COOKIES_FILE` に指定します。

X動画の取得、音声抽出、文字起こしは Kurage Voice Pro の既存実装
`kuragevp/backend/pipeline.py` を再利用します。X動画は `yt-dlp` ではなく
fxtwitter API から動画URLを取得し、Whisperは KurageVP と同じ CUDA/float16 構成で動かします。

ブログ/記事URLはHTML本文抽出を優先し、失敗時は Jina Reader を使います。PDFは
`pdftotext` を優先し、Python PDFライブラリがある環境ではそれも利用します。X投稿は
fxtwitter の本文・メディア情報を見て、動画付きなら従来の動画分析、動画なしなら記事/本文分析として扱います。

## Notes

OpenMontage is AGPL-3.0 and is kept as a reference clone. Kurage Montage does not vendor OpenMontage code into the product path; it implements a Kurage-specific reference-video explainer workflow.

Useful OpenMontage ideas adopted in kmontage:

- Save intermediate artifacts instead of jumping straight to rendering.
- Separate reference analysis, scene planning, final script, and render QA.
- Keep source-basis notes so generated videos can be audited for faithfulness.
- Treat generic, abstract summaries as failures for reference-video jobs.
