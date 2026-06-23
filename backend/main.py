from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
STORAGE_DIR = ROOT / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
KURAGE_API = os.environ.get("KURAGE_API", "http://127.0.0.1:18303").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.0.14:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
YTDLP_BIN = os.environ.get("YTDLP_BIN", "yt-dlp")
YTDLP_COOKIES_FILE = os.environ.get("KMONTAGE_YTDLP_COOKIES_FILE", "")
YTDLP_COOKIES_BROWSER = os.environ.get("KMONTAGE_YTDLP_COOKIES_BROWSER", "")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
TRANSCRIBE_PYTHON = os.environ.get("KMONTAGE_TRANSCRIBE_PYTHON", "/home/kojima/work/kuragevp/.venv/bin/python")
TRANSCRIBE_MODEL = os.environ.get("KMONTAGE_TRANSCRIBE_MODEL", "small")
ENABLE_TRANSCRIBE = os.environ.get("KMONTAGE_ENABLE_TRANSCRIBE", "1").lower() not in {"0", "false", "no"}

app = FastAPI(title="Kurage Montage", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CreateJobRequest(BaseModel):
    url: str
    vtuber_mode: bool = True
    video_style: str = "ai_avatar_explainer"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def load_job(job_id: str) -> dict[str, Any] | None:
    p = job_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_job(job_id: str, **kwargs: Any) -> dict[str, Any]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    p = job_path(job_id)
    data: dict[str, Any] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    data.update(kwargs)
    data["updated_at"] = now()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return data


def run_cmd(args: list[str], *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


def ollama_generate_url() -> str:
    if OLLAMA_URL.endswith("/api/generate"):
        return OLLAMA_URL
    return f"{OLLAMA_URL}/api/generate"


def ytdlp_args(*extra: str) -> list[str]:
    args = [YTDLP_BIN]
    if YTDLP_COOKIES_FILE:
        args.extend(["--cookies", YTDLP_COOKIES_FILE])
    elif YTDLP_COOKIES_BROWSER:
        args.extend(["--cookies-from-browser", YTDLP_COOKIES_BROWSER])
    args.extend(extra)
    return args


def url_kind(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if host in {"x.com", "twitter.com", "mobile.twitter.com"} or host.endswith(".x.com") or host.endswith(".twitter.com"):
        return "x"
    return "video"


def extract_vtt_text(vtt: str) -> str:
    lines = []
    seen = set()
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return " ".join(lines)


def extract_json3_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return ""
    lines = []
    seen = set()
    for event in data.get("events", []):
        segs = event.get("segs") or []
        text = "".join(str(seg.get("utf8") or "") for seg in segs)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(text)
    return " ".join(lines)


def fetch_metadata(url: str, job_dir: Path) -> dict[str, Any]:
    result = run_cmd(ytdlp_args("-J", "--no-playlist", url), timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "yt-dlp metadata failed")
    meta = json.loads(result.stdout)
    (job_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def captions_from_metadata(meta: dict[str, Any]) -> str:
    pools = []
    for key in ("subtitles", "automatic_captions"):
        value = meta.get(key) or {}
        if isinstance(value, dict):
            pools.append(value)
    preferred = ["ja", "ja-JP", "en", "en-US", "en-orig"]
    for pool in pools:
        keys = preferred + [k for k in pool.keys() if k not in preferred]
        for lang in keys:
            tracks = pool.get(lang) or []
            if not isinstance(tracks, list):
                continue
            for track in tracks:
                ext = str(track.get("ext") or "").lower()
                track_url = track.get("url")
                if not track_url or ext not in {"vtt", "srv3", "ttml", "json3"}:
                    continue
                try:
                    res = requests.get(track_url, timeout=30)
                    res.raise_for_status()
                    if ext == "vtt":
                        text = extract_vtt_text(res.text)
                    elif ext == "json3":
                        text = extract_json3_text(res.text)
                    else:
                        text = re.sub(r"<[^>]+>", " ", res.text)
                        text = re.sub(r"[{}\[\]\\\"_:,0-9]+", " ", text)
                        text = re.sub(r"\s+", " ", text).strip()
                    if len(text) >= 40:
                        return text
                except Exception:
                    continue
    return ""


def download_reference_video(url: str, job_dir: Path) -> Path | None:
    media_dir = job_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(media_dir / "source.%(ext)s")
    result = run_cmd(ytdlp_args(
        "--no-playlist",
        "--max-filesize", os.environ.get("KMONTAGE_MAX_FILESIZE", "300m"),
        "-f", "bv*+ba/best",
        "-o", out_tmpl,
        url,
    ), timeout=600)
    if result.returncode != 0:
        (job_dir / "download_warning.log").write_text((result.stderr or result.stdout)[-4000:], encoding="utf-8")
        return None
    files = sorted(media_dir.glob("source.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def media_duration(path: Path) -> float:
    result = run_cmd([FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def transcribe_video(video_path: Path, job_dir: Path) -> str:
    if not ENABLE_TRANSCRIBE or not Path(TRANSCRIBE_PYTHON).exists():
        return ""
    audio = job_dir / "media" / "audio.wav"
    result = run_cmd([FFMPEG_BIN, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(audio)], timeout=300)
    if result.returncode != 0 or not audio.exists():
        return ""
    worker = Path(__file__).with_name("transcribe_worker.py")
    result = run_cmd([TRANSCRIBE_PYTHON, str(worker), str(audio), TRANSCRIBE_MODEL], timeout=1200)
    if result.returncode != 0:
        (job_dir / "transcribe_error.log").write_text(result.stderr + "\n" + result.stdout, encoding="utf-8")
        return ""
    try:
        data = json.loads(result.stdout)
    except Exception:
        return ""
    (job_dir / "transcript.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(data.get("text") or "").strip()


def build_analysis_prompt(url: str, kind: str, meta: dict[str, Any], transcript: str) -> str:
    title = meta.get("title") or "参照動画"
    description = meta.get("description") or ""
    uploader = meta.get("uploader") or meta.get("channel") or ""
    duration = meta.get("duration") or ""
    return f"""次の参照動画を理解し、日本語ショート動画に再構成するための制作プランを作ってください。

URL: {url}
種類: {kind}
タイトル: {title}
投稿者: {uploader}
長さ: {duration}秒
説明文:
{description[:1200]}

文字起こし/字幕:
{transcript[:5000]}

要件:
- 日本語で出力
- 元動画を丸ごと転載するのではなく、要点解説・考察にする
- 視聴者が最初の3秒で何の話かわかるフックを作る
- 60〜120秒の縦型ショート向け
- Kurage VTuberが話す想定
- 誇張しすぎず、元動画で確認できる範囲と考察を分ける

JSONのみで返してください。
形式:
{{
  "title": "日本語タイトル",
  "summary": "要点の短い要約",
  "key_points": ["要点1", "要点2", "要点3", "要点4"],
  "script_outline": ["導入", "要点", "背景", "考察", "まとめ"],
  "kurage_content": "Kurageに渡す動画生成用本文。600〜1200字。"
}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end+1]
    return json.loads(text)


def analyze_reference(url: str, kind: str, meta: dict[str, Any], transcript: str, job_dir: Path) -> dict[str, Any]:
    prompt = build_analysis_prompt(url, kind, meta, transcript)
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 4096}}
    res = requests.post(ollama_generate_url(), json=payload, timeout=300)
    res.raise_for_status()
    response = res.json().get("response") or ""
    (job_dir / "analysis_response.txt").write_text(response, encoding="utf-8")
    try:
        analysis = parse_json_object(response)
    except Exception:
        title = meta.get("title") or "参照動画の要点解説"
        base = transcript or meta.get("description") or title
        analysis = {
            "title": f"{title} 要点解説",
            "summary": base[:300],
            "key_points": [line.strip() for line in re.split(r"[。\n]", base) if line.strip()][:4],
            "script_outline": ["参照動画のテーマ", "重要ポイント", "背景", "Kurageの考察", "まとめ"],
            "kurage_content": base[:1200],
        }
    return analysis


def enqueue_kurage(job_id: str, url: str, kind: str, analysis: dict[str, Any], vtuber_mode: bool, video_style: str) -> str:
    title = str(analysis.get("title") or "参照動画の要点解説").strip()
    key_points = analysis.get("key_points") or []
    outline = analysis.get("script_outline") or []
    content = str(analysis.get("kurage_content") or analysis.get("summary") or title).strip()
    content = (
        f"参照動画URL: {url}\n"
        f"入力タイプ: {kind}\n\n"
        f"要約: {analysis.get('summary','')}\n\n"
        f"重要ポイント:\n" + "\n".join(f"- {p}" for p in key_points) + "\n\n"
        f"構成案:\n" + "\n".join(f"- {p}" for p in outline) + "\n\n"
        f"本文:\n{content}"
    )
    payload = {
        "title": title,
        "news_items": [{"title": title, "content": content, "url": url, "source_name": "Kurage Montage"}],
        "vtuber_mode": vtuber_mode,
        "video_style": video_style,
    }
    res = requests.post(f"{KURAGE_API}/generate_from_news", json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    kurage_job_id = data.get("job_id")
    if not kurage_job_id:
        raise RuntimeError(f"Kurage did not return job_id: {data}")
    save_job(job_id, kurage_job_id=kurage_job_id, kurage_url=f"https://kurage.exbridge.jp/kuragev.php?id={kurage_job_id}")
    return kurage_job_id


def refresh_from_kurage(job: dict[str, Any]) -> dict[str, Any]:
    kurage_job_id = job.get("kurage_job_id")
    if not kurage_job_id:
        return job
    try:
        res = requests.get(f"{KURAGE_API}/status/{kurage_job_id}", timeout=20)
        if res.status_code != 200:
            return job
        status = res.json()
    except Exception:
        return job
    updates: dict[str, Any] = {
        "kurage_status": status.get("status"),
        "kurage_progress": status.get("progress"),
        "kurage_title": status.get("title"),
        "kurage_script": status.get("script"),
    }
    if status.get("status") == "done":
        updates.update({
            "status": "done",
            "progress": 100,
            "video_url": f"https://kurage.exbridge.jp/kuragev.php?id={kurage_job_id}",
            "kurage_video_endpoint": f"{KURAGE_API}/video/{kurage_job_id}",
        })
    elif status.get("status") == "error":
        updates.update({"status": "error", "error": status.get("error") or "Kurage generation failed"})
    else:
        updates.update({"status": "generating", "progress": 55 + int(status.get("progress") or 0) // 3})
    return save_job(job["id"], **updates)


def process_job(job_id: str) -> None:
    job = load_job(job_id) or {}
    url = job.get("url") or ""
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        kind = url_kind(url)
        save_job(job_id, status="analyzing", progress=10, kind=kind)
        meta = fetch_metadata(url, job_dir)
        transcript = captions_from_metadata(meta)
        save_job(job_id, progress=25, source_title=meta.get("title"), source_uploader=meta.get("uploader") or meta.get("channel"), transcript_preview=transcript[:500])

        video_path = None
        if len(transcript) < 80:
            save_job(job_id, status="downloading", progress=30)
            video_path = download_reference_video(url, job_dir)
            if video_path:
                save_job(job_id, reference_video=str(video_path), reference_duration=media_duration(video_path))
                save_job(job_id, status="transcribing", progress=38)
                transcript = transcribe_video(video_path, job_dir) or transcript
        save_job(job_id, transcript_preview=transcript[:1000])

        if not transcript and not (meta.get("description") or meta.get("title")):
            raise RuntimeError("動画内容を解析できませんでした。認証済みブラウザ録画などの取得経路が必要です。")

        save_job(job_id, status="planning", progress=45)
        analysis = analyze_reference(url, kind, meta, transcript, job_dir)
        save_job(job_id, analysis=analysis, title=analysis.get("title"), summary=analysis.get("summary"), script_outline=analysis.get("script_outline") or [])

        save_job(job_id, status="generating", progress=55)
        kurage_job_id = enqueue_kurage(job_id, url, kind, analysis, bool(job.get("vtuber_mode", True)), str(job.get("video_style") or "ai_avatar_explainer"))
        save_job(job_id, kurage_job_id=kurage_job_id, status="generating", progress=60)

        deadline = time.time() + 3600
        while time.time() < deadline:
            latest = refresh_from_kurage(load_job(job_id) or {"id": job_id})
            if latest.get("status") in {"done", "error"}:
                return
            time.sleep(15)
        raise RuntimeError("Kurage video generation timed out")
    except Exception as exc:
        save_job(job_id, status="error", error=str(exc), progress=100)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "kmontage", "time": now(), "kurage_api": KURAGE_API, "ollama_url": OLLAMA_URL}


@app.post("/api/jobs")
def create_job(req: CreateJobRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    if url_kind(url) not in {"x", "youtube"}:
        raise HTTPException(status_code=400, detail="X URL または YouTube URL を入力してください")
    job_id = uuid.uuid4().hex[:16]
    save_job(job_id, id=job_id, url=url, status="queued", progress=0, vtuber_mode=req.vtuber_mode, video_style=req.video_style, created_at=now())
    thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return refresh_from_kurage(job)


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
            jobs.append(job)
        except Exception:
            pass
    return {"ok": True, "jobs": jobs}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    kurage_job_id = job.get("kurage_job_id")
    if kurage_job_id:
        try:
            requests.delete(f"{KURAGE_API}/jobs/{kurage_job_id}", timeout=20)
        except Exception:
            pass
    job_file = job_path(job_id)
    job_dir = JOBS_DIR / job_id
    if job_file.exists():
        job_file.unlink()
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"ok": True, "job_id": job_id, "kurage_job_id": kurage_job_id}
