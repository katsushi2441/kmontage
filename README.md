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
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 18305
```

Open:

```text
http://localhost:18305/
```

## Environment

```bash
KURAGE_API=http://127.0.0.1:18303
OLLAMA_URL=http://192.168.0.14:11434
OLLAMA_MODEL=gemma4:e4b
KMONTAGE_TRANSCRIBE_PYTHON=/home/kojima/work/kuragevp/.venv/bin/python
KMONTAGE_TRANSCRIBE_MODEL=small
KMONTAGE_YTDLP_COOKIES_BROWSER=chrome
# or
KMONTAGE_YTDLP_COOKIES_FILE=/path/to/cookies.txt
```

X動画は未ログイン状態だと取得できないことがあります。その場合は、認証済みブラウザを使っている環境で
`KMONTAGE_YTDLP_COOKIES_BROWSER=chrome` か、書き出したCookieファイルを
`KMONTAGE_YTDLP_COOKIES_FILE` に指定します。

## Notes

OpenMontage is AGPL-3.0 and is kept as a reference clone. Kurage Montage does not vendor OpenMontage code into the product path; it implements a Kurage-specific reference-video explainer workflow.
