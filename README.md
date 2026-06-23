# Kurage Montage

Kurage Montage (`kmontage`) turns a reference X or YouTube video URL into a Japanese short explainer video through Kurage.

## Structure

- `backend/`: FastAPI app for reference video analysis and Kurage job orchestration.
- `static/`: single-page UI.
- `storage/jobs/`: local job metadata. Downloaded media is ignored.
- `OpenMontage/`: upstream reference OSS clone, intentionally ignored by Git.

## Flow

1. User inputs an X URL or YouTube URL.
2. `yt-dlp` reads metadata, captions, and when needed downloads the reference video.
3. Captions or Whisper transcription are summarized by Ollama into a Japanese production plan.
4. The generated summary is sent to Kurage `/generate_from_news` with VTuber mode enabled.
5. The resulting video appears in `kuragev.php` because Kurage owns the final job JSON/video.

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

## Notes

OpenMontage is AGPL-3.0 and is kept as a reference clone. Kurage Montage does not vendor OpenMontage code into the product path; it implements a Kurage-specific reference-video explainer workflow.
