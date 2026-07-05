from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.parse import urlunparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
KURAGE_API = os.environ.get("KURAGE_API", "http://127.0.0.1:18303").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.0.14:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b-it-qat")
OLLAMA_TIMEOUT = int(os.environ.get("KMONTAGE_OLLAMA_TIMEOUT", "360"))
OLLAMA_NUM_PREDICT = int(os.environ.get("KMONTAGE_OLLAMA_NUM_PREDICT", "8192"))
USE_RQDB4AI_OLLAMA = os.environ.get("KMONTAGE_USE_RQDB4AI_OLLAMA", "1").lower() not in {"0", "false", "no"}
RQDB4AI_OLLAMA_QUEUE_CLASS = os.environ.get("KMONTAGE_RQDB4AI_OLLAMA_QUEUE_CLASS", "web")
RQDB4AI_OLLAMA_TIMEOUT = int(os.environ.get("KMONTAGE_RQDB4AI_OLLAMA_TIMEOUT", "900"))
YTDLP_BIN = os.environ.get("YTDLP_BIN", "yt-dlp")
YTDLP_COOKIES_FILE = os.environ.get("KMONTAGE_YTDLP_COOKIES_FILE", "")
YTDLP_COOKIES_BROWSER = os.environ.get("KMONTAGE_YTDLP_COOKIES_BROWSER", "")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
ENABLE_TRANSCRIBE = os.environ.get("KMONTAGE_ENABLE_TRANSCRIBE", "1").lower() not in {"0", "false", "no"}
KURAGEVP_BACKEND_DIR = Path(os.environ.get("KURAGEVP_BACKEND_DIR", "/home/kojima/work/kuragevp/backend"))
KAGENTREACH_ROOT = Path(os.environ.get("KAGENTREACH_ROOT", "/home/kojima/work/kagentreach"))
KAGENTREACH_NEWS_OPINION_SCRIPT = Path(
    os.environ.get("KAGENTREACH_NEWS_OPINION_SCRIPT", str(KAGENTREACH_ROOT / "scripts" / "news-opinion-research.py"))
)

app = FastAPI(title="Kurage Montage", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CREATE_JOB_LOCK = threading.Lock()
ACTIVE_JOB_STATUSES = {
    "queued",
    "analyzing",
    "downloading",
    "transcribing",
    "researching",
    "planning",
    "generating",
}


class CreateJobRequest(BaseModel):
    url: str
    vtuber_mode: bool = True
    video_style: str = "ai_avatar_explainer"
    mode: str = "summary"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_source_url(url: str) -> str:
    text = (url or "").strip()
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = re.sub(r"/+$", "", parsed.path or "/")
    host = netloc.split("@")[-1].split(":", 1)[0]
    if host in {"x.com", "twitter.com", "mobile.twitter.com"}:
        match = re.search(r"/(?:i/)?status(?:es)?/(\d+)", path)
        if match:
            return f"https://x.com/i/status/{match.group(1)}"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def is_active_job(job: dict[str, Any]) -> bool:
    return str(job.get("status") or "").lower() in ACTIVE_JOB_STATUSES


def job_sort_timestamp(job: dict[str, Any], fallback: float = 0.0) -> float:
    for key in ("updated_at", "created_at"):
        text = str(job.get(key) or "").strip()
        if not text:
            continue
        try:
            return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
    return fallback


def normalize_job_progress(job: dict[str, Any]) -> dict[str, Any]:
    """Keep completed jobs visually complete even if older JSON kept stale progress."""
    status = str(job.get("status") or "").strip().lower()
    kurage_status = str(job.get("kurage_status") or "").strip().lower()
    if status == "done" or kurage_status == "done":
        job["status"] = "done"
        job["progress"] = 100
        if job.get("kurage_job_id"):
            job["kurage_status"] = "done"
            job["kurage_progress"] = 100
    return job


def find_active_job_for_url(url: str, mode: str) -> dict[str, Any] | None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    normalized = normalize_source_url(url)
    matches: list[dict[str, Any]] = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(job.get("mode") or "summary") != mode:
            continue
        if not is_active_job(job):
            continue
        if normalize_source_url(str(job.get("url") or "")) != normalized:
            continue
        if job.get("kurage_job_id") and job.get("status") not in {"done", "error"}:
            try:
                job = refresh_from_kurage(job)
            except Exception:
                pass
        if is_active_job(job):
            matches.append(job)
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return matches[0]


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def load_job(job_id: str) -> dict[str, Any] | None:
    p = job_path(job_id)
    if not p.exists():
        return None
    return normalize_job_progress(json.loads(p.read_text(encoding="utf-8")))


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


def replace_job(job_id: str, data: dict[str, Any]) -> dict[str, Any]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    p = job_path(job_id)
    data = dict(data)
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


def kuragevp_pipeline():
    backend_dir = str(KURAGEVP_BACKEND_DIR)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import pipeline  # type: ignore

    return pipeline


def url_kind(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if host in {"x.com", "twitter.com", "mobile.twitter.com"} or host.endswith(".x.com") or host.endswith(".twitter.com"):
        return "x"
    return "article"


def is_video_kind(kind: str) -> bool:
    return kind in {"x", "youtube", "video"}


def source_label(kind: str) -> str:
    if kind == "pdf":
        return "PDF資料"
    if kind == "article":
        return "記事/ブログ"
    if kind == "x":
        return "X投稿/記事"
    return "参照動画"


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if len(line) <= 2:
            continue
        key = line[:180]
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines).strip()


def jina_reader_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url
    return "https://r.jina.ai/" + url


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


def extract_m3u8_vtt_text(playlist: str, playlist_url: str) -> str:
    """X/Twitter may expose subtitles as an HLS playlist that points to VTT."""
    candidates = []
    for raw in playlist.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ".vtt" in line.lower():
            candidates.append(urljoin(playlist_url, line))
    for url in candidates:
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            text = extract_vtt_text(res.text)
            if len(text) >= 40:
                return text
        except Exception:
            continue
    return ""


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


def fetch_x_metadata(url: str, job_dir: Path) -> dict[str, Any]:
    """Fetch X/Twitter metadata through KurageVP's fxtwitter path.

    yt-dlp's Twitter extractor can fail with transient guest-token errors.
    KurageVP already uses fxtwitter for X videos, so kmontage should not fail
    before it reaches that working media-download/transcription path.
    """
    tweet_id = ""
    try:
        tweet_id = kuragevp_pipeline().extract_tweet_id(url)
        data = kuragevp_pipeline().fetch_fxtwitter(tweet_id)
    except Exception as exc:
        (job_dir / "metadata_warning.log").write_text(f"fxtwitter metadata failed: {exc}", encoding="utf-8")
        data = {}
    tweet = data.get("tweet") if isinstance(data.get("tweet"), dict) else {}
    author = tweet.get("author") if isinstance(tweet.get("author"), dict) else {}
    media = tweet.get("media") if isinstance(tweet.get("media"), dict) else {}
    article = tweet.get("article") if isinstance(tweet.get("article"), dict) else {}
    article_title = str(article.get("title") or "").strip()
    article_preview = str(article.get("preview_text") or "").strip()
    article_text = extract_fxtwitter_article_text(article)
    text = str(tweet.get("text") or "").strip()
    if not text:
        raw_text = tweet.get("raw_text") if isinstance(tweet.get("raw_text"), dict) else {}
        text = str(raw_text.get("text") or "").strip()
    description = clean_extracted_text("\n\n".join(p for p in [text, article_preview, article_text] if p))
    author_name = str(author.get("name") or author.get("screen_name") or author.get("username") or "X").strip()
    uploader = "@" + str(author.get("screen_name") or author.get("username") or "X").lstrip("@")
    title = article_title or (text[:90] if text else f"X投稿 {tweet_id}".strip())
    meta = {
        "id": tweet_id,
        "webpage_url": url,
        "original_url": url,
        "extractor": "fxtwitter",
        "title": title or "X投稿",
        "description": description,
        "uploader": uploader,
        "channel": author_name,
        "duration": max([float(v.get("duration") or 0) for v in (media.get("videos") or [])] or [0]),
        "media_count": len(media.get("videos") or []) + len(media.get("photos") or []),
        "has_video_media": bool(media.get("videos")),
        "has_article": bool(article_text or article_title),
        "subtitles": {},
        "automatic_captions": {},
    }
    (job_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if data:
        (job_dir / "fxtwitter.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def extract_fxtwitter_article_text(article: dict[str, Any]) -> str:
    parts = []
    title = str(article.get("title") or "").strip()
    preview = str(article.get("preview_text") or "").strip()
    if title:
        parts.append(title)
    if preview:
        parts.append(preview)
    content = article.get("content") if isinstance(article.get("content"), dict) else {}
    blocks = content.get("blocks") if isinstance(content.get("blocks"), list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        block_type = str(block.get("type") or "")
        if not text or block_type == "atomic":
            continue
        parts.append(text)
    return clean_extracted_text("\n".join(parts))


def fetch_reference_metadata(url: str, kind: str, job_dir: Path) -> dict[str, Any]:
    try:
        return fetch_metadata(url, job_dir)
    except Exception as exc:
        (job_dir / "metadata_warning.log").write_text(str(exc), encoding="utf-8")
        if kind == "x":
            meta = fetch_x_metadata(url, job_dir)
            meta["metadata_fallback_reason"] = str(exc)
            (job_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return meta
        raise


def html_metadata_and_text(url: str, job_dir: Path) -> tuple[dict[str, Any], str]:
    headers = {"User-Agent": "Mozilla/5.0 KurageMontage/1.0"}
    res = requests.get(url, headers=headers, timeout=40)
    res.raise_for_status()
    content_type = res.headers.get("content-type", "")
    if "pdf" in content_type.lower():
        pdf_path = job_dir / "source.pdf"
        pdf_path.write_bytes(res.content)
        meta = {
            "id": Path(urlparse(url).path).name or "pdf",
            "webpage_url": url,
            "original_url": url,
            "extractor": "pdf",
            "title": Path(urlparse(url).path).name or "PDF資料",
            "description": "",
            "uploader": urlparse(url).netloc,
            "channel": urlparse(url).netloc,
            "duration": 0,
        }
        return meta, extract_pdf_text(pdf_path, job_dir)

    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
    if og_title and og_title.get("content"):
        title = str(og_title.get("content")).strip()
    description = ""
    desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if desc_tag and desc_tag.get("content"):
        description = str(desc_tag.get("content")).strip()
    article = soup.find("article") or soup.find("main") or soup.body or soup
    headings = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])[:8])
    paragraphs = [p.get_text(" ", strip=True) for p in article.find_all(["p", "li", "blockquote"])]
    text = clean_extracted_text("\n".join([title, description, headings, *paragraphs]))
    meta = {
        "id": Path(urlparse(url).path).name or "article",
        "webpage_url": url,
        "original_url": url,
        "extractor": "html",
        "title": title or url,
        "description": description,
        "uploader": urlparse(url).netloc,
        "channel": urlparse(url).netloc,
        "duration": 0,
    }
    return meta, text


def jina_text(url: str, job_dir: Path) -> tuple[dict[str, Any], str]:
    reader_url = jina_reader_url(url)
    res = requests.get(reader_url, timeout=60)
    res.raise_for_status()
    raw = res.text
    (job_dir / "jina_reader.txt").write_text(raw, encoding="utf-8")
    title = ""
    m = re.search(r"^Title:\s*(.+)$", raw, flags=re.M)
    if m:
        title = m.group(1).strip()
    markdown = raw
    marker = "Markdown Content:"
    if marker in raw:
        markdown = raw.split(marker, 1)[1]
    text = clean_extracted_text(markdown)
    meta = {
        "id": Path(urlparse(url).path).name or "article",
        "webpage_url": url,
        "original_url": url,
        "extractor": "jina_reader",
        "title": title or url,
        "description": text[:500],
        "uploader": urlparse(url).netloc,
        "channel": urlparse(url).netloc,
        "duration": 0,
    }
    return meta, text


def extract_pdf_text(pdf_path: Path, job_dir: Path) -> str:
    txt_path = job_dir / "source.txt"
    result = run_cmd(["pdftotext", "-layout", str(pdf_path), str(txt_path)], timeout=120)
    if result.returncode == 0 and txt_path.exists():
        text = clean_extracted_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
        if len(text) >= 80:
            return text
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:40])
        return clean_extracted_text(text)
    except Exception as exc:
        (job_dir / "pdf_extract_warning.log").write_text(str(exc), encoding="utf-8")
    return ""


def fetch_pdf_document(url: str, job_dir: Path) -> tuple[dict[str, Any], str]:
    headers = {"User-Agent": "Mozilla/5.0 KurageMontage/1.0"}
    res = requests.get(url, headers=headers, timeout=60)
    res.raise_for_status()
    pdf_path = job_dir / "source.pdf"
    pdf_path.write_bytes(res.content)
    title = Path(urlparse(url).path).name or "PDF資料"
    meta = {
        "id": title,
        "webpage_url": url,
        "original_url": url,
        "extractor": "pdf",
        "title": title,
        "description": "",
        "uploader": urlparse(url).netloc,
        "channel": urlparse(url).netloc,
        "duration": 0,
    }
    return meta, extract_pdf_text(pdf_path, job_dir)


def fetch_x_article_text(url: str, job_dir: Path) -> tuple[dict[str, Any], str]:
    meta = fetch_x_metadata(url, job_dir)
    text_parts = [str(meta.get("description") or "").strip()]
    result = run_cmd(["twitter", "article", url], timeout=60)
    if result.returncode == 0 and result.stdout.strip():
        (job_dir / "twitter_article.txt").write_text(result.stdout, encoding="utf-8")
        text_parts.append(result.stdout.strip())
    else:
        (job_dir / "twitter_article_warning.log").write_text((result.stderr or result.stdout)[-4000:], encoding="utf-8")
    if sum(len(p) for p in text_parts) < 500:
        try:
            _, text = jina_text(url, job_dir)
            text_parts.append(text)
        except Exception as exc:
            (job_dir / "jina_warning.log").write_text(str(exc), encoding="utf-8")
    article_text = clean_extracted_text("\n\n".join(p for p in text_parts if p))
    if article_text:
        meta["description"] = article_text[:1200]
    return meta, article_text


def fetch_document_source(url: str, kind: str, job_dir: Path) -> tuple[dict[str, Any], str]:
    try:
        if kind == "pdf":
            return fetch_pdf_document(url, job_dir)
        if kind == "x":
            return fetch_x_article_text(url, job_dir)
        return html_metadata_and_text(url, job_dir)
    except Exception as exc:
        (job_dir / "document_fetch_warning.log").write_text(str(exc), encoding="utf-8")
        if kind == "x":
            meta = fetch_x_metadata(url, job_dir)
            text = clean_extracted_text(str(meta.get("description") or ""))
            return meta, text
        if kind in {"article", "pdf"}:
            return jina_text(url, job_dir)
        raise


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
                        if res.text.lstrip().startswith("#EXTM3U"):
                            text = extract_m3u8_vtt_text(res.text, track_url)
                        else:
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


def download_reference_video(url: str, job_dir: Path, kind: str) -> Path | None:
    media_dir = job_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    if kind == "x":
        try:
            return kuragevp_pipeline().download_video(url, media_dir)
        except Exception as exc:
            (job_dir / "download_warning.log").write_text(str(exc), encoding="utf-8")
            return None

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
    if not ENABLE_TRANSCRIBE:
        return ""
    try:
        pipeline = kuragevp_pipeline()
        media_dir = job_dir / "media"
        audio = pipeline.extract_audio(video_path, media_dir)
        source_srt, source_txt = pipeline.transcribe_audio(audio, media_dir, "auto")
        text = Path(source_txt).read_text(encoding="utf-8").strip()
        (job_dir / "transcript.json").write_text(json.dumps({
            "ok": True,
            "source": "kuragevp.pipeline.transcribe_audio",
            "srt": str(source_srt),
            "txt": str(source_txt),
            "text": text,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return text
    except Exception as exc:
        (job_dir / "transcribe_error.log").write_text(str(exc), encoding="utf-8")
        return ""


def build_analysis_prompt(url: str, kind: str, meta: dict[str, Any], transcript: str) -> str:
    label = source_label(kind)
    title = meta.get("title") or label
    description = meta.get("description") or ""
    uploader = meta.get("uploader") or meta.get("channel") or ""
    duration = meta.get("duration") or ""
    transcript_excerpt = compact_reference_text(transcript, 7000)
    return f"""次の{label}を分析し、日本語ショート動画に再構成してください。

これは「一般論の解説」ではありません。元資料の中心主張、具体的な数字、ツール、手順、注意点、読者にとっての意味を忠実に抽出してください。
元資料にない話は入れないでください。考察を入れる場合は「考察」と明示してください。

URL: {url}
種類: {kind}
タイトル: {title}
投稿者: {uploader}
長さ: {duration}秒
説明文:
{description[:1200]}

本文/文字起こし/字幕:
{transcript_excerpt}

要件:
- 日本語で出力
- 元資料を丸ごと転載するのではなく、要点解説・考察にする
- 視聴者が最初の3秒で何の話かわかるフックを作る
- 60〜120秒の縦型ショート向け、12シーン、各10秒程度
- Kurage VTuberが話す想定
- 抽象論で終わらせない。「何を、どの順番で、なぜやるのか」を入れる
- 金額、再生数、RPM、制作費、期間、投稿頻度などの数字を優先して残す
- ツール名や作業工程がある場合は残す
- 著作権、コピー、シャドウバン、低品質量産などの注意点があれば残す
- 競合分析、台本作成、画像生成、音声生成、編集、外注化などの手順があれば残す
- 最終台本は、元動画の要点を忠実に圧縮した内容にする

OpenMontageの設計思想を参考に、次の中間成果物を明示してください。
- reference_analysis: 元資料の事実、数字、手順、注意点、バズった/読む価値がある構造
- scene_plan: どの要点をどの順で見せるか
- script: Kurageがそのまま動画化できる12シーン台本

JSONのみで返してください。
形式:
{{
  "reference_analysis": {{
    "title": "元資料の要点タイトル",
    "core_claim": "元資料が主張している中心命題",
    "evidence_numbers": ["30日で3300ドル", "制作費20ドル", "140K再生で700ドル売上"],
    "workflow_steps": ["手順1", "手順2", "手順3"],
    "tools_or_methods": ["Claude", "AI voiceover", "CapCut"],
    "risks": ["注意点1", "注意点2"],
    "why_it_went_viral": ["理由1", "理由2"]
  }},
  "scene_plan": {{
    "title": "日本語動画タイトル",
    "target_duration": 120,
    "scenes": [
      {{"index":0,"role":"hook","source_basis":"元動画の根拠","message":"このシーンで伝える要点"}}
    ]
  }},
  "script": {{
    "title": "日本語動画タイトル",
    "scenes": [
      {{"index":0,"narration":"日本語ナレーション。具体的な数字や手順を含める。","image_prompt":"English vertical 9:16 explainer visual under 100 chars","duration":10}}
    ]
  }},
  "qa": {{
    "concrete_facts_used": ["台本に入れた具体事実"],
    "omitted_topics": ["短尺化のため省略した要素"],
    "faithfulness_note": "元動画への忠実性の説明"
  }}
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        salvaged = salvage_script_analysis(text)
        if salvaged:
            return salvaged
        raise


def _json_string_value(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw.replace('\\"', '"').replace("\\n", "\n")


def salvage_script_analysis(text: str) -> dict[str, Any] | None:
    """Recover concrete script scenes from slightly malformed LLM JSON.

    Local LLMs sometimes emit useful scene objects but miss a comma or the final
    closing brackets. We only recover explicit narration/image_prompt pairs; we
    never invent generic filler here.
    """
    title = "参照動画の要点解説"
    title_match = re.search(r'"script"\s*:\s*\{.*?"title"\s*:\s*"((?:\\.|[^"])*)"', text, flags=re.S)
    if not title_match:
        title_match = re.search(r'"title"\s*:\s*"((?:\\.|[^"])*)"', text, flags=re.S)
    if title_match:
        title = _json_string_value(title_match.group(1)).strip() or title

    scenes: list[dict[str, Any]] = []
    scene_pattern = re.compile(
        r'"narration"\s*:\s*"((?:\\.|[^"])*)"\s*,\s*'
        r'"image_prompt"\s*:\s*"((?:\\.|[^"])*)"\s*,\s*'
        r'"duration"\s*:\s*(\d+)',
        flags=re.S,
    )
    for match in scene_pattern.finditer(text):
        narration = _json_string_value(match.group(1)).strip()
        image_prompt = _json_string_value(match.group(2)).strip()
        if not narration:
            continue
        scenes.append({
            "index": len(scenes),
            "narration": narration,
            "image_prompt": image_prompt or "clean Japanese vertical explainer, data cards, 9:16",
            "duration": int(match.group(3) or 8),
        })
        if len(scenes) >= 12:
            break

    if len(scenes) < 6:
        return None

    evidence = extract_source_numbers("\n".join([title] + [s["narration"] for s in scenes]))
    workflow = [s["narration"] for s in scenes[: min(5, len(scenes))]]
    return {
        "reference_analysis": {
            "title": title,
            "core_claim": scenes[0]["narration"],
            "evidence_numbers": evidence,
            "workflow_steps": workflow,
            "tools_or_methods": [],
            "risks": [],
            "why_it_went_viral": [],
        },
        "scene_plan": {
            "title": title,
            "target_duration": sum(int(s.get("duration") or 8) for s in scenes),
            "scenes": [
                {"index": s["index"], "role": "salvaged_scene", "source_basis": "malformed_llm_json", "message": s["narration"]}
                for s in scenes
            ],
        },
        "script": {"title": title, "scenes": scenes},
        "qa": {
            "concrete_facts_used": evidence,
            "omitted_topics": [],
            "faithfulness_note": "Recovered from explicit LLM scene objects after malformed JSON.",
        },
    }


def compact_reference_text(text: str, limit: int = 7000) -> str:
    """Keep enough source detail without sending an unbounded X article to Ollama."""
    text = clean_extracted_text(text or "")
    if len(text) <= limit:
        return text
    numeric_lines = []
    for line in text.splitlines():
        if re.search(r"\d|\\$|ドル|円|万|%|RPM|Claude|HyperFrames|Google|AI|YouTube|TikTok|Threads", line, flags=re.I):
            numeric_lines.append(line)
    middle = clean_extracted_text("\n".join(numeric_lines))[: max(1000, limit // 4)]
    head = text[: max(1800, limit // 2)]
    tail = text[-max(1000, limit // 5):]
    return clean_extracted_text(
        f"{head}\n\n--- extracted important lines ---\n{middle}\n\n--- source tail ---\n{tail}"
    )[:limit]


def ollama_generate(prompt: str, job_dir: Path, label: str, *, temperature: float = 0.1, num_predict: int | None = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict or OLLAMA_NUM_PREDICT},
    }
    try:
        if USE_RQDB4AI_OLLAMA:
            response = rqdb4ai_ollama_generate(prompt, job_dir, label, temperature=temperature, num_predict=num_predict)
        else:
            res = requests.post(ollama_generate_url(), json=payload, timeout=OLLAMA_TIMEOUT)
            res.raise_for_status()
            response = res.json().get("response") or ""
        (job_dir / f"{label}_response.txt").write_text(response, encoding="utf-8")
        return response
    except Exception as exc:
        (job_dir / f"{label}_error.log").write_text(str(exc), encoding="utf-8")
        raise


def rqdb4ai_ollama_generate(prompt: str, job_dir: Path, label: str, *, temperature: float = 0.1, num_predict: int | None = None) -> str:
    helper = ROOT / "scripts" / "rqdb4ai_ollama_generate.py"
    prompt_file = job_dir / f"{label}_prompt.txt"
    result_file = job_dir / f"{label}_rqdb4ai_result.json"
    prompt_file.write_text(prompt, encoding="utf-8")
    args = [
        sys.executable if "rq" in sys.modules else "/usr/bin/python3",
        str(helper),
        "--prompt-file", str(prompt_file),
        "--result-file", str(result_file),
        "--ollama-url", OLLAMA_URL,
        "--model", OLLAMA_MODEL,
        "--temperature", str(temperature),
        "--num-predict", str(num_predict or OLLAMA_NUM_PREDICT),
        "--queue-class", RQDB4AI_OLLAMA_QUEUE_CLASS,
        "--timeout", str(RQDB4AI_OLLAMA_TIMEOUT),
        "--source", "web_online",
    ]
    proc = subprocess.run(args, cwd=str(ROOT), text=True, capture_output=True, timeout=RQDB4AI_OLLAMA_TIMEOUT + 60)
    if proc.stdout.strip():
        (job_dir / f"{label}_rqdb4ai_stdout.log").write_text(proc.stdout[-8000:], encoding="utf-8")
    if proc.stderr.strip():
        (job_dir / f"{label}_rqdb4ai_stderr.log").write_text(proc.stderr[-8000:], encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"rqdb4ai Ollama job failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-1200:]}")
    data = json.loads(result_file.read_text(encoding="utf-8"))
    response = data.get("response") or ""
    if not response:
        raise RuntimeError(f"rqdb4ai Ollama job returned empty response: {data}")
    return response


def analyze_reference(url: str, kind: str, meta: dict[str, Any], transcript: str, job_dir: Path) -> dict[str, Any]:
    prompt = build_analysis_prompt(url, kind, meta, transcript)
    try:
        response = ollama_generate(prompt, job_dir, "analysis", temperature=0.1)
        analysis = parse_json_object(response)
    except Exception as exc:
        (job_dir / "analysis_primary_error.log").write_text(str(exc), encoding="utf-8")
        analysis = retry_reference_analysis(url, kind, meta, transcript, job_dir)
    analysis = normalize_reference_analysis(analysis, meta, transcript)
    analysis = expand_short_but_specific_script(analysis, meta, transcript)
    if needs_japanese_repair(analysis):
        repaired = repair_analysis_to_japanese(url, kind, meta, transcript, analysis, job_dir)
        analysis = normalize_reference_analysis(repaired, meta, transcript)
        analysis = expand_short_but_specific_script(analysis, meta, transcript)
    if needs_japanese_repair(analysis):
        (job_dir / "japanese_quality_error.json").write_text(json.dumps({
            "reason": "script_is_not_japanese_enough",
            "title": (analysis.get("script") or {}).get("title"),
            "scene_count": len(((analysis.get("script") or {}).get("scenes") or [])),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("日本語ショート台本の生成に失敗しました。英語台本のままKurageへ送信しないため停止しました。")
    issues = script_quality_issues(analysis, meta, transcript)
    if issues:
        (job_dir / "script_quality_repair_reason.json").write_text(json.dumps({
            "reason": "retry_quality_repair",
            "issues": issues,
            "title": (analysis.get("script") or {}).get("title"),
            "scene_count": len(((analysis.get("script") or {}).get("scenes") or [])),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        # Feed the gate's findings back to the model: the concrete numbers/terms it
        # dropped, so the repair re-includes them instead of staying generic.
        focus_facts = list(dict.fromkeys(
            quality_source_numbers(analysis, meta, transcript)
            + extract_source_terms(quality_source_text(meta, transcript))[:8]
        ))
        repaired = repair_analysis_to_japanese(url, kind, meta, transcript, analysis, job_dir, must_include=focus_facts)
        analysis = normalize_reference_analysis(repaired, meta, transcript)
        analysis = expand_short_but_specific_script(analysis, meta, transcript)
        issues = script_quality_issues(analysis, meta, transcript)
    if issues:
        (job_dir / "script_quality_error.json").write_text(json.dumps({
            "reason": "script_is_too_generic_or_unfaithful",
            "issues": issues,
            "title": (analysis.get("script") or {}).get("title"),
            "scene_count": len(((analysis.get("script") or {}).get("scenes") or [])),
            "source_title": meta.get("title"),
            "transcript_chars": len(transcript or ""),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("元資料に忠実な具体台本を作れませんでした。汎用的なインチキ動画を生成しないため停止しました。詳細は script_quality_error.json を確認してください。")
    write_openmontage_artifacts(job_dir, analysis)
    return analysis


def build_news_opinion_query(meta: dict[str, Any], transcript: str) -> str:
    title = str(meta.get("title") or "").strip()
    if title:
        return title[:140]
    text = clean_extracted_text(transcript or str(meta.get("description") or ""))
    first = re.split(r"[。.!?\n]", text)[0].strip()
    return first[:140]


def collect_news_opinions(url: str, meta: dict[str, Any], transcript: str, job_dir: Path) -> dict[str, Any]:
    if not KAGENTREACH_NEWS_OPINION_SCRIPT.exists():
        raise RuntimeError(f"Kurage AgentReach opinion collector not found: {KAGENTREACH_NEWS_OPINION_SCRIPT}")
    out_file = job_dir / "news_opinions.json"
    query = build_news_opinion_query(meta, transcript)
    args = [
        sys.executable,
        str(KAGENTREACH_NEWS_OPINION_SCRIPT),
        "--url", url,
        "--title", str(meta.get("title") or ""),
        "--query", query,
        "--out", str(out_file),
        "--limit", "8",
    ]
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("news.yahoo.co.jp") and "/comments" in parsed.path:
        # Yahoo comments are already the primary source for kmontagenews.
        # Avoid blocking the hourly worker on optional browser-use X search.
        args.append("--skip-x")
    proc = subprocess.run(args, cwd=str(KAGENTREACH_ROOT), text=True, capture_output=True, timeout=1200)
    if proc.stdout.strip():
        (job_dir / "kagentreach_news_stdout.log").write_text(proc.stdout[-12000:], encoding="utf-8")
    if proc.stderr.strip():
        (job_dir / "kagentreach_news_stderr.log").write_text(proc.stderr[-12000:], encoding="utf-8")
    data: dict[str, Any] = {}
    if out_file.exists():
        data = json.loads(out_file.read_text(encoding="utf-8"))
    elif proc.stdout.strip():
        data = json.loads(proc.stdout)
    if proc.returncode != 0 and not data.get("opinion_points"):
        detail = (proc.stderr or proc.stdout or "unknown error")[-1200:]
        raise RuntimeError(f"Kurage AgentReachでニュース反応を収集できませんでした: {detail}")
    points = data.get("opinion_points") if isinstance(data.get("opinion_points"), list) else []
    if len(points) < 3:
        raise RuntimeError("ニュースに関する意見候補が不足しています。汎用的な反応紹介動画を生成しないため停止しました。")
    if url_kind(url) == "x":
        x_replies = ((data.get("sources") or {}).get("x_replies") or [])
        x_reply_points = [
            p for p in points
            if isinstance(p, dict) and str(p.get("platform") or "") == "Xリプライ"
        ]
        if len(x_replies) < 3 or len(x_reply_points) < 3:
            errors = data.get("errors") if isinstance(data.get("errors"), list) else []
            detail = " / ".join(str(e) for e in errors if "x_replies" in str(e))[:800]
            raise RuntimeError(
                "X投稿のリプライ取得が不足しています。X投稿では、いいねが多いリプライを主役にするため、"
                f"認証済みtwitter-cliで3件以上のリプライ取得が必要です。{detail}"
            )
    return data


def build_news_opinion_prompt(url: str, kind: str, meta: dict[str, Any], transcript: str, opinions: dict[str, Any]) -> str:
    title = str(meta.get("title") or source_label(kind)).strip()
    author_info = ""
    if kind == "x":
        author_name = str(meta.get("channel") or "").strip()
        author_handle = str(meta.get("uploader") or "").strip()
        author_info = f"\nX投稿者: {author_name} {author_handle}".strip()
    article_text = compact_reference_text("\n".join([
        str(meta.get("title") or ""),
        str(meta.get("description") or ""),
        transcript or "",
    ]), 5200)
    opinion_json = json.dumps({
        "query": opinions.get("query"),
        "summary": opinions.get("summary"),
        "opinion_points": opinions.get("opinion_points") or [],
        "errors": opinions.get("errors") or [],
    }, ensure_ascii=False, indent=2)[:7000]
    is_x = kind == "x"
    x_extra_rules = ""
    if is_x:
        x_extra_rules = """
X投稿URLの場合の最重要ルール:
- 主役は元投稿への「Xリプライ」。YahooコメントやWeb検索より、いいね数が多いXリプライを最優先に扱う。
- 冒頭1〜2シーンで元投稿の内容を短く紹介し、残りは「いいね上位リプライの論点整理」に使う。
- 「Xリプライでは」「いいね上位のリプライでは」「別のリプライでは」のように、必ずリプライ由来だと分かる言い方にする。
- リプライ本文を丸読みせず、要点に整理する。ただし、賛成・反対・懸念・実務目線などの違いは潰さない。
- いいね数がある場合は「いいねが多いリプライでは」のように重みを伝える。
- 投稿者名や人物名を勝手に漢字化しない。本文やリプライにない実名は推測せず、「投稿者」「本田さん」「{meta.get("channel") or "投稿者"}」のように根拠のある呼び方だけ使う。
"""
    return f"""あなたはKurage Montage Newsの編集者です。
ニュースURLの本文と、Kurage AgentReachが収集したYahooコメント・X・YouTube・ブログ/Webの反応をもとに、Kurageアバターが紹介する2分程度の日本語ショート動画台本JSONを作ってください。

重要:
- ニュース本文と収集された反応だけを根拠にする。知らない事実を足さない。
- URLをアルファベットで読み上げない。
- 「みんなが言っています」のような断定は禁止。必ず「Yahooコメントでは」「Xでは」「YouTubeでは」「ブログでは」「一部では」のように出所を分ける。
- Yahooコメントを使う場合は、共感数が多いコメントを「共感数の多い意見」として扱い、コメント本文を丸読みせず要点に整理する。
- Xリプライを使う場合は、いいね数が多いリプライを「いいね上位の反応」として扱い、リプライ本文を丸読みせず要点に整理する。
- ニュース紹介は短めにする。冒頭1〜2シーン、全体の20%以下まで。
- 動画の主役は「みんなの意見」。12シーン中8シーン以上を、共感数上位コメントやWeb/YouTube/Xの反応整理に使う。
- 1シーン目はサムネを意識する。ニュースの争点と、共感数上位の代表意見は narration と後段のHyperFramesテロップで伝える。画像生成AIに日本語文字を描かせない。
- 1シーン目の narration は短く、ニュースの争点 + 代表意見を45〜80字程度で言い切る。
- image_prompt は明るいWhite Studioで、黒背景禁止。冒頭は thumbnail-like vertical composition, blank headline panels without text, news issue + top public opinion visual metaphor を必ず含める。
- image_prompt には no text, no letters, no numbers, blank cards only を入れる。カードやUIは空欄にし、読ませたい文字はHyperFramesで重ねる前提にする。
- ニュース紹介だけで終わらず、賛成・懸念・実務目線・今後の見方など、複数の意見を整理する。
- 台本は自然な日本語。英語原文をそのまま貼らない。
- 60〜120秒、12シーン、各8〜10秒程度。
- image_promptだけ英語でよい。明るいWhite Studio、Kurage avatar explainerの映像指示にする。
- 十分な根拠がない場合は {{"error":"insufficient_reaction_detail","reason":"理由"}} を返す。汎用台本で埋めない。
- JSONのみで返す。
{x_extra_rules}

ニュースURL: {url}
種類: {kind}
ニュースタイトル: {title}
{author_info}

ニュース本文/取得内容:
{article_text}

Kurage AgentReach 収集結果:
{opinion_json}

返すJSON形式:
{{
  "reference_analysis": {{
    "title": "日本語タイトル",
    "core_claim": "このニュースと反応を一文で要約",
    "news_summary": ["ニュース本文に基づく具体要点"],
    "opinion_summary": ["Yahooコメント/X/YouTube/ブログなどの反応要点"],
    "evidence_numbers": ["数字や固有名詞"],
    "workflow_steps": ["視聴者が理解する流れ"],
    "tools_or_methods": ["関連ツールや方法があれば"],
    "risks": ["懸念や注意点"],
    "why_it_matters": ["なぜ重要か"]
  }},
  "scene_plan": {{
    "title": "日本語動画タイトル",
    "target_duration": 110,
    "scenes": [{{"index":0,"role":"hook","source_basis":"ニュース本文または収集反応の根拠","message":"日本語の要点"}}]
  }},
  "script": {{
    "title": "日本語動画タイトル",
    "scenes": [{{"index":0,"narration":"日本語ナレーション","image_prompt":"bright white studio vertical news explainer with cute Kurage avatar, clean cards, 9:16","duration":9}}]
  }},
  "qa": {{
    "concrete_facts_used": ["台本に入れた具体事実"],
    "reaction_sources_used": ["Yahooコメント/X/YouTube/Webのどれを使ったか"],
    "omitted_topics": [],
    "faithfulness_note": "ニュース本文と反応に忠実である説明"
  }}
}}
"""


def news_opinion_quality_issues(analysis: dict[str, Any], opinions: dict[str, Any]) -> list[str]:
    script = analysis.get("script") if isinstance(analysis.get("script"), dict) else {}
    scenes = script.get("scenes") if isinstance(script.get("scenes"), list) else []
    scene_plan = analysis.get("scene_plan") if isinstance(analysis.get("scene_plan"), dict) else {}
    plan_scenes = scene_plan.get("scenes") if isinstance(scene_plan.get("scenes"), list) else []
    narrations = [str(s.get("narration") or "") for s in scenes if isinstance(s, dict)]
    joined = "\n".join([str(script.get("title") or "")] + narrations)
    points = opinions.get("opinion_points") if isinstance(opinions.get("opinion_points"), list) else []
    issues: list[str] = []
    if len(scenes) < 10:
        issues.append(f"scene_count_too_low:{len(scenes)}")
    if japanese_chars(joined) < 240 or any(text_looks_english(n) for n in narrations):
        issues.append("script_is_not_japanese_enough")
    if not any(word in joined for word in ["意見", "反応", "Yahoo", "コメント", "Xでは", "YouTube", "ブログ", "Web", "一部では", "懸念"]):
        issues.append("missing_reaction_framing")
    yahoo_points = [
        p for p in points
        if isinstance(p, dict) and str(p.get("platform") or "") == "Yahooコメント"
    ]
    x_reply_points = [
        p for p in points
        if isinstance(p, dict) and str(p.get("platform") or "") == "Xリプライ"
    ]
    source_text = "\n".join(str(p.get("point") or "") for p in points)
    source_terms = extract_source_terms(source_text)[:16]
    reaction_scene_count = 0
    for idx, n in enumerate(narrations):
        script_scene = scenes[idx] if idx < len(scenes) and isinstance(scenes[idx], dict) else {}
        plan_scene = plan_scenes[idx] if idx < len(plan_scenes) and isinstance(plan_scenes[idx], dict) else {}
        scene_context = "\n".join([
            n,
            str(script_scene.get("role") or ""),
            str(script_scene.get("source_basis") or ""),
            str(plan_scene.get("role") or ""),
            str(plan_scene.get("source_basis") or ""),
            str(plan_scene.get("message") or ""),
        ])
        explicit_reaction = any(
            word in scene_context
            for word in [
                "意見", "反応", "共感", "Yahoo", "コメント", "一部では", "懸念", "賛成", "批判",
                "指摘", "視点", "反論", "疑問", "不満", "期待", "問われています",
                "opinion", "reaction", "x_reply", "news_opinion",
            ]
        )
        source_based = any(term and term.lower() in scene_context.lower() for term in source_terms)
        if explicit_reaction or source_based:
            reaction_scene_count += 1
    if len(yahoo_points) >= 3 and reaction_scene_count < 6:
        issues.append(f"too_few_opinion_scenes:{reaction_scene_count}")
    if len(x_reply_points) >= 3:
        x_reply_markers = [
            "Xリプライ", "リプライ", "いいね", "反応", "賛成", "反対", "懸念", "指摘",
            "期待", "ワクワク", "本気", "覚悟", "参謀", "提案", "ファン", "ユーモア",
            "興味", "議論", "支持",
        ]
        x_reply_scene_count = sum(
            1 for n in narrations
            if any(word in n for word in x_reply_markers)
        )
        if x_reply_scene_count < 6:
            issues.append(f"too_few_x_reply_scenes:{x_reply_scene_count}")
        if not any(word in joined for word in ["Xリプライ", "リプライ", "いいね"]):
            issues.append("missing_x_reply_framing")
    first_prompt = str(scenes[0].get("image_prompt") or "").lower() if scenes else ""
    if scenes and not any(word in first_prompt for word in ["thumbnail", "headline", "card", "text"]):
        issues.append("opening_visual_not_thumbnail_like")
    if len(points) >= 3:
        used = 0
        for term in extract_source_terms(source_text)[:10]:
            if term.lower() in joined.lower():
                used += 1
        yahoo_reactions_reflected = bool(yahoo_points) and any(word in joined for word in ["Yahoo", "コメント", "共感"])
        x_reactions_reflected = bool(x_reply_points) and any(word in joined for word in ["Xリプライ", "リプライ", "いいね", "反応"])
        if used < 1 and not yahoo_reactions_reflected and not x_reactions_reflected and len(clean_extracted_text(source_text)) >= 300:
            issues.append("reactions_not_reflected")
    return issues


def shorten_news_narration(text: str, limit: int = 86) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    candidates = re.split(r"(?<=[。！？])", text)
    out = ""
    for part in candidates:
        if not part:
            continue
        if len(out + part) <= limit:
            out += part
        elif out:
            break
    if len(out) >= 28:
        return out.strip()
    cut = text[:limit].rstrip("、。 ")
    if "、" in cut:
        cut = cut.rsplit("、", 1)[0]
    return (cut or text[:limit]).strip() + "。"


def compact_news_opinion_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Keep news-opinion videos around two minutes and avoid Kurage re-splitting scenes."""
    script = analysis.get("script") if isinstance(analysis.get("script"), dict) else {}
    scenes = script.get("scenes") if isinstance(script.get("scenes"), list) else []
    if not scenes:
        return analysis

    def is_opinion(scene: dict[str, Any]) -> bool:
        text = " ".join(str(scene.get(k) or "") for k in ["narration", "role", "source_basis"])
        return any(word in text for word in ["意見", "反応", "共感", "Yahoo", "コメント", "Xリプライ", "リプライ", "いいね", "懸念", "批判", "不満", "賛否"])

    first = scenes[:1]
    opinion_scenes = [s for s in scenes[1:] if isinstance(s, dict) and is_opinion(s)]
    other_scenes = [s for s in scenes[1:] if isinstance(s, dict) and not is_opinion(s)]
    selected = first + opinion_scenes[:10]
    for scene in other_scenes:
        if len(selected) >= 12:
            break
        selected.append(scene)
    if len(selected) < 10:
        selected = [s for s in scenes if isinstance(s, dict)][:12]
    else:
        selected = selected[:12]

    cleaned: list[dict[str, Any]] = []
    for scene in selected:
        narration = shorten_news_narration(str(scene.get("narration") or ""), 82 if len(cleaned) else 78)
        if not narration:
            continue
        prompt = str(scene.get("image_prompt") or "bright white studio vertical news opinion explainer, Kurage avatar, clean cards, 9:16").strip()
        if not cleaned and "thumbnail" not in prompt.lower():
            prompt = "thumbnail-like vertical composition, blank headline panels without text, news issue + top public opinion visual metaphor, bright white studio, Kurage avatar explainer, no text, no letters, no numbers"
        cleaned.append({
            "index": len(cleaned),
            "narration": narration,
            "image_prompt": prompt[:220],
            "duration": 9 if len(cleaned) == 0 else 8,
        })

    script["scenes"] = cleaned[:12]
    script["title"] = str(script.get("title") or analysis.get("reference_analysis", {}).get("title") or "ニュース反応まとめ")[:70]
    analysis["script"] = script
    plan = analysis.get("scene_plan") if isinstance(analysis.get("scene_plan"), dict) else {}
    plan["target_duration"] = sum(int(s.get("duration") or 8) for s in script["scenes"])
    plan["scenes"] = [
        {
            "index": s["index"],
            "role": "news_opinion",
            "source_basis": "news/opinion",
            "message": s["narration"],
        }
        for s in script["scenes"]
    ]
    analysis["scene_plan"] = plan
    return analysis


def analyze_news_opinions(url: str, kind: str, meta: dict[str, Any], transcript: str, opinions: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    prompt = build_news_opinion_prompt(url, kind, meta, transcript, opinions)
    try:
        response = ollama_generate(prompt, job_dir, "news_opinion_analysis", temperature=0.08)
        analysis = parse_json_object(response)
    except Exception as exc:
        (job_dir / "news_opinion_analysis_error.log").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"ニュース反応台本の生成に失敗しました。原因: {exc}")
    if analysis.get("error"):
        raise RuntimeError(f"ニュース反応台本を生成できません: {analysis.get('error')} {analysis.get('reason')}")
    analysis = normalize_reference_analysis(analysis, meta, transcript)
    analysis = compact_news_opinion_analysis(analysis)
    issues = news_opinion_quality_issues(analysis, opinions)
    if issues:
        (job_dir / "news_opinion_quality_error.json").write_text(json.dumps({
            "reason": "news_opinion_script_quality_failed",
            "issues": issues,
            "opinion_count": len(opinions.get("opinion_points") or []),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("ニュース反応を反映した具体的な日本語台本を作れませんでした。インチキ動画を生成しないため停止しました。")
    write_openmontage_artifacts(job_dir, analysis)
    return analysis


def retry_reference_analysis(url: str, kind: str, meta: dict[str, Any], transcript: str, job_dir: Path) -> dict[str, Any]:
    """Retry with a smaller prompt instead of falling back to generic filler."""
    compact = compact_reference_text(transcript or str(meta.get("description") or ""), 4200)
    title = meta.get("title") or source_label(kind)
    prompt = f"""次の元資料から、日本語ショート動画の台本JSONだけを作ってください。

重要: 一般論は禁止。元資料にある数字、商品、手順、ツール、注意点だけを使う。
元資料に十分な情報がない場合は、JSONで {{"error":"insufficient_source_detail","reason":"理由"}} を返す。

URL: {url}
タイトル: {title}
本文/文字起こし:
{compact}

必須:
- 日本語タイトル
- 12シーン
- 各 narration は元資料の具体情報を含む
- 金額、期間、ツール名、商品名、手順があれば必ず残す
- image_prompt だけ英語でよい

JSON形式:
{{
  "reference_analysis": {{
    "title": "日本語タイトル",
    "core_claim": "中心主張",
    "evidence_numbers": ["具体的な数字"],
    "workflow_steps": ["具体的な手順"],
    "tools_or_methods": ["ツールや方法"],
    "risks": ["注意点"],
    "why_it_went_viral": ["伸びた理由"]
  }},
  "scene_plan": {{
    "title": "日本語動画タイトル",
    "target_duration": 100,
    "scenes": [{{"index":0,"role":"hook","source_basis":"元資料の根拠","message":"日本語の要点"}}]
  }},
  "script": {{
    "title": "日本語動画タイトル",
    "scenes": [{{"index":0,"narration":"日本語ナレーション","image_prompt":"English vertical 9:16 explainer visual","duration":8}}]
  }},
  "qa": {{
    "concrete_facts_used": ["台本に入れた具体事実"],
    "omitted_topics": [],
    "faithfulness_note": "忠実性の説明"
  }}
}}
"""
    try:
        response = ollama_generate(prompt, job_dir, "analysis_retry", temperature=0.05, num_predict=8192)
        analysis = parse_json_object(response)
        if analysis.get("error"):
            raise RuntimeError(f"analysis_retry_failed: {analysis.get('error')} {analysis.get('reason')}")
        return analysis
    except Exception as exc:
        (job_dir / "analysis_retry_error.log").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"LLM解析に失敗しました。汎用フォールバック動画は生成しません。原因: {exc}")


def japanese_chars(text: str) -> int:
    return sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")


def ascii_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isascii() and ch.isalpha())


def text_looks_english(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    jp = japanese_chars(text)
    en = ascii_letters(text)
    return en >= 35 and en > max(12, jp * 2)


def needs_japanese_repair(analysis: dict[str, Any]) -> bool:
    script = analysis.get("script") if isinstance(analysis.get("script"), dict) else {}
    scenes = script.get("scenes") if isinstance(script.get("scenes"), list) else []
    title = str(script.get("title") or "")
    narrations = [str(s.get("narration") or "") for s in scenes if isinstance(s, dict)]
    joined = "\n".join([title] + narrations)
    if len(scenes) < 6:
        return True
    if text_looks_english(title):
        return True
    if any(text_looks_english(n) for n in narrations):
        return True
    # A Japanese short should have enough Japanese signal across the script.
    return japanese_chars(joined) < 180


GENERIC_TITLE_PATTERNS = [
    "参照資料から学ぶ",
    "AI活用の要点",
    "参照動画の要点",
    "要点解説",
]

GENERIC_NARRATION_PATTERNS = [
    "短い教材や投稿から",
    "すばやく学べる",
    "何を作るのか、どのツールを使うのか",
    "単なる紹介で終わらせず",
    "小さく試して結果を見ながら",
    "初心者でも最初の一歩",
    "内容が薄くなったり",
    "元資料の根拠、数字、具体例",
    "フック、手順、注意点",
    "再現性のある仕組み",
    "AIを使って学び、作り、改善するサイクル",
    "自分のプロジェクトで試す",
]


def extract_source_numbers(text: str) -> list[str]:
    unit_pattern = (
        r"ドル|円|万円|万|億|%|％|割|つ|個|枚|件|社|人|回|"
        r"ステップ|項目|条件|モード|ツール|ブロック|パート|章|"
        r"RPM|views?|再生|日|週|週間|ヶ月|カ月|年|時間|分|"
        r"steps?|parts?|blocks?|items?|conditions?|tools?|modes?|"
        r"sales?|visitors?|months?|years?"
    )
    raw = re.findall(
        rf"\b\d+\s?B\b|\b\d+\s?-\s?\d+\b|(?:\$|¥)?\d[\d,.]*(?:\s?(?:{unit_pattern}))?",
        text or "",
        flags=re.I,
    )
    large_percent_tokens = {
        re.sub(r"[^\d]", "", v)
        for v in raw
        if re.search(r"[%％]", v) and int(re.sub(r"[^\d]", "", v) or "0") >= 50
    }
    cleaned = []
    seen = set()
    for value in raw:
        v = re.sub(r"\s+", " ", value).strip().rstrip(".,")
        if len(v) <= 1:
            continue
        # Whisper can occasionally turn "95% faster" into "9% faster" later in
        # the transcript. If strong percentage claims are already present, do
        # not force a one-digit percentage into the final script.
        if re.fullmatch(r"\d[%％]", v) and large_percent_tokens:
            continue
        # Bare one/two digit list markers such as 01., 02., 10. are section
        # numbers, not concrete evidence that must appear in a short script.
        has_unit_or_currency = bool(re.search(rf"(?:\$|¥|{unit_pattern})", v, flags=re.I))
        has_compact_model_size = bool(re.fullmatch(r"\d+\s?B", v, flags=re.I))
        has_around_the_clock = bool(re.fullmatch(r"\d+\s?-\s?\d+", v))
        if not (has_unit_or_currency or has_compact_model_size or has_around_the_clock) and re.fullmatch(r"\d{1,2}", v):
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(v)
    return cleaned[:20]


def extract_source_terms(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9.+_-]{2,}\b|\b(?:Claude|Kittle|Etsy|Google Drive|ListingView|Flux|Nano Banana|PNG|PDF|ADK|ReAct|YouTube|CapCut|HyperFrames)\b", text or "", flags=re.I)
    seen = set()
    terms = []
    stop = {
        "the", "and", "for", "this", "that", "with", "can", "now", "run",
        "free", "using", "latest", "makes", "model", "today", "want", "show",
        "you", "how", "forever", "new", "around", "from", "into", "something",
        "actually", "use", "one", "single", "click", "just", "like",
    }
    for term in candidates:
        t = term.strip()
        key = t.lower()
        if key in seen or key in stop:
            continue
        seen.add(key)
        terms.append(t)
    return terms[:24]


def script_quality_issues(analysis: dict[str, Any], meta: dict[str, Any], transcript: str) -> list[str]:
    script = analysis.get("script") if isinstance(analysis.get("script"), dict) else {}
    scenes = script.get("scenes") if isinstance(script.get("scenes"), list) else []
    title = str(script.get("title") or "")
    narrations = [str(s.get("narration") or "") for s in scenes if isinstance(s, dict)]
    reference = analysis.get("reference_analysis") if isinstance(analysis.get("reference_analysis"), dict) else {}
    evidence = reference.get("evidence_numbers") if isinstance(reference.get("evidence_numbers"), list) else []
    workflow = reference.get("workflow_steps") if isinstance(reference.get("workflow_steps"), list) else []
    joined = "\n".join([title] + narrations + [str(x) for x in evidence] + [str(x) for x in workflow])
    source_text = quality_source_text(meta, transcript)
    source_numbers = quality_source_numbers(analysis, meta, transcript)
    source_terms = extract_source_terms(source_text)
    issues: list[str] = []

    if len(scenes) < 10:
        issues.append(f"scene_count_too_low:{len(scenes)}")
    if any(p in title for p in GENERIC_TITLE_PATTERNS) and not any(term.lower() in title.lower() for term in source_terms[:8]):
        issues.append("generic_title")
    generic_hits = sum(1 for line in narrations for p in GENERIC_NARRATION_PATTERNS if p in line)
    if generic_hits >= 3:
        issues.append(f"generic_narration_phrases:{generic_hits}")
    if len(source_text) >= 1200 and len(joined) < 500:
        issues.append(f"script_too_short_for_source:{len(joined)}")

    if len(source_numbers) >= 4:
        matched = sum(1 for n in source_numbers if normalize_number_token(n) and normalize_number_token(n) in normalize_number_token(joined))
        if matched < 3:
            issues.append(f"missing_source_numbers:{matched}/{len(source_numbers)}")
    if len(source_numbers) >= 4 and len(evidence) < 3:
        issues.append("reference_analysis_missing_evidence")
    if len(source_text) >= 1200 and len(workflow) < 3:
        issues.append("reference_analysis_missing_workflow")
    return issues


def quality_source_text(meta: dict[str, Any], transcript: str) -> str:
    """Use source content, not YouTube promotional description noise, for QA."""
    title = str(meta.get("title") or "")
    transcript = transcript or ""
    if len(clean_extracted_text(transcript)) >= 300:
        return clean_extracted_text("\n".join([title, transcript]))
    return clean_extracted_text("\n".join([title, str(meta.get("description") or ""), transcript]))


def quality_source_numbers(analysis: dict[str, Any], meta: dict[str, Any], transcript: str) -> list[str]:
    """Prefer LLM-extracted evidence numbers over noisy YouTube descriptions."""
    reference = analysis.get("reference_analysis") if isinstance(analysis.get("reference_analysis"), dict) else {}
    evidence = reference.get("evidence_numbers") if isinstance(reference.get("evidence_numbers"), list) else []
    evidence_numbers = extract_source_numbers("\n".join(str(x) for x in evidence))
    if len(evidence_numbers) >= 3:
        return evidence_numbers[:12]
    return extract_source_numbers(quality_source_text(meta, transcript))[:12]


def split_narration_for_short_scene(text: str) -> list[str]:
    """Split one faithful source-based scene into two short narration beats."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    if len(sentences) >= 2:
        while len(sentences) >= 2 and len(sentences[0]) < 24:
            sentences[1] = sentences[0] + sentences[1]
            sentences.pop(0)
        mid = max(1, len(sentences) // 2)
        return ["".join(sentences[:mid]).strip(), "".join(sentences[mid:]).strip()]
    for sep in ["。", "、", "。"]:
        if sep in text and len(text) > 44:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) >= 2:
                mid = max(1, len(parts) // 2)
                first = sep.join(parts[:mid]).strip()
                second = sep.join(parts[mid:]).strip()
                if sep == "。" and first and not first.endswith("。"):
                    first += "。"
                if sep == "。" and second and not second.endswith("。"):
                    second += "。"
                return [first, second]
    if len(text) > 70:
        mid = len(text) // 2
        return [text[:mid].rstrip("、。 "), text[mid:].lstrip("、。 ")]
    return [text]


def expand_short_but_specific_script(analysis: dict[str, Any], meta: dict[str, Any], transcript: str) -> dict[str, Any]:
    """Turn a good but too-short script into 10-12 faithful scenes.

    Local LLMs often understand the source correctly but return six scenes for
    five-step videos. That should not be treated as an unfaithful script. Split
    each existing source-based narration into smaller beats instead of falling
    back to generic filler or publishing fewer scenes.
    """
    script = analysis.get("script") if isinstance(analysis.get("script"), dict) else {}
    scenes = script.get("scenes") if isinstance(script.get("scenes"), list) else []
    if not (5 <= len(scenes) < 10):
        return analysis

    title = str(script.get("title") or meta.get("title") or "参照動画の要点解説")
    source_text = quality_source_text(meta, transcript)
    source_numbers = quality_source_numbers(analysis, meta, transcript)
    joined = "\n".join(str(s.get("narration") or "") for s in scenes if isinstance(s, dict))
    matched_numbers = sum(1 for n in source_numbers if normalize_number_token(n) and normalize_number_token(n) in normalize_number_token(joined))
    reference = analysis.get("reference_analysis") if isinstance(analysis.get("reference_analysis"), dict) else {}
    workflow = reference.get("workflow_steps") if isinstance(reference.get("workflow_steps"), list) else []
    evidence = reference.get("evidence_numbers") if isinstance(reference.get("evidence_numbers"), list) else []
    evidence_joined = "\n".join(str(x) for x in evidence + workflow)
    matched_in_analysis = sum(
        1 for n in source_numbers
        if normalize_number_token(n) and normalize_number_token(n) in normalize_number_token(joined + "\n" + evidence_joined)
    )
    if len(source_numbers) >= 4 and matched_numbers < 2 and matched_in_analysis < 3:
        return analysis
    if len(source_text) >= 1200 and len(workflow) < 3:
        return analysis

    expanded: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        narration = str(scene.get("narration") or "").strip()
        if not narration:
            continue
        pieces = split_narration_for_short_scene(narration)
        if len(pieces) == 1 and len(expanded) + (len(scenes) - len(expanded)) < 10:
            pieces = [pieces[0]]
        for piece in pieces:
            if not piece:
                continue
            expanded.append({
                "index": len(expanded),
                "narration": piece[:150],
                "image_prompt": str(scene.get("image_prompt") or "clean Japanese vertical explainer, data cards, 9:16").strip()[:180],
                "duration": max(6, min(9, int(scene.get("duration") or 8))),
            })
            if len(expanded) >= 12:
                break
        if len(expanded) >= 12:
            break

    if len(expanded) < 10:
        plan = analysis.get("scene_plan") if isinstance(analysis.get("scene_plan"), dict) else {}
        plan_scenes = plan.get("scenes") if isinstance(plan.get("scenes"), list) else []
        for plan_scene in plan_scenes:
            if not isinstance(plan_scene, dict):
                continue
            message = str(plan_scene.get("message") or "").strip()
            if not message or any(message == s.get("narration") for s in expanded):
                continue
            expanded.append({
                "index": len(expanded),
                "narration": message[:150],
                "image_prompt": "clean Japanese vertical explainer, source-based data cards, 9:16",
                "duration": 8,
            })
            if len(expanded) >= 12:
                break

    if len(expanded) >= 10:
        analysis["script"] = {"title": title[:70], "scenes": expanded[:12]}
        plan = analysis.get("scene_plan") if isinstance(analysis.get("scene_plan"), dict) else {}
        plan["title"] = plan.get("title") or title
        plan["target_duration"] = sum(int(s.get("duration") or 8) for s in expanded[:12])
        plan["scenes"] = [
            {
                "index": s["index"],
                "role": "source_split",
                "source_basis": "split from source-based LLM scene",
                "message": s["narration"],
            }
            for s in expanded[:12]
        ]
        analysis["scene_plan"] = plan
    return analysis


def normalize_number_token(text: str) -> str:
    return re.sub(r"[^\d]", "", text or "")


def build_japanese_repair_prompt(url: str, kind: str, meta: dict[str, Any], transcript: str, analysis: dict[str, Any], must_include: list[str] | None = None) -> str:
    label = source_label(kind)
    title = meta.get("title") or label
    uploader = meta.get("uploader") or meta.get("channel") or ""
    description = meta.get("description") or ""
    transcript_excerpt = compact_reference_text(transcript, 5000)
    previous = json.dumps(analysis, ensure_ascii=False)[:8000]
    must_block = ""
    if must_include:
        joined = " / ".join(str(x).strip() for x in must_include[:14] if str(x).strip())
        if joined:
            must_block = (
                "\n最重要: 前回は元資料を一般論に薄めてしまいました。次の具体情報を必ず"
                "タイトルか各ナレーションに織り込み、汎用的な言い換えで省略しないこと:\n"
                f"{joined}\n"
            )
    return f"""次の{label}から、日本語ショート解説動画の台本を作り直してください。

前回の解析結果に英語タイトル・英語ナレーション・シーン不足が混ざりました。今回は必ず日本語で、12シーンの完成台本にしてください。

URL: {url}
種類: {kind}
タイトル: {title}
投稿者: {uploader}
説明文:
{description[:1200]}

本文/文字起こし/字幕:
{transcript_excerpt}

前回解析結果（参考。英語のまま使わず、日本語へ要約し直す）:
{previous}
{must_block}
必須条件:
- title、reference_analysis、scene_plan.message、script.scenes[].narration はすべて自然な日本語
- 英語字幕をそのまま貼り付けない。固有名詞とツール名以外は日本語にする
- 元資料の数字、手順、収益、ツール、注意点は忠実に残す
- 60〜120秒の縦型ショート向け、12シーン、各8〜10秒程度
- image_prompt だけは英語でよい
- JSONのみで返す

形式:
{{
  "reference_analysis": {{
    "title": "日本語タイトル",
    "core_claim": "元資料の中心主張を日本語で1文",
    "evidence_numbers": ["具体的な数字"],
    "workflow_steps": ["日本語の手順"],
    "tools_or_methods": ["ツールや方法"],
    "risks": ["注意点"],
    "why_it_went_viral": ["伸びた理由"]
  }},
  "scene_plan": {{
    "title": "日本語動画タイトル",
    "target_duration": 100,
    "scenes": [{{"index":0,"role":"hook","source_basis":"根拠","message":"日本語の要点"}}]
  }},
  "script": {{
    "title": "日本語動画タイトル",
    "scenes": [{{"index":0,"narration":"日本語ナレーション","image_prompt":"English vertical 9:16 explainer visual","duration":8}}]
  }},
  "qa": {{
    "concrete_facts_used": ["台本に入れた具体事実"],
    "omitted_topics": ["省略した要素"],
    "faithfulness_note": "忠実性の説明"
  }}
}}
"""


def repair_analysis_to_japanese(url: str, kind: str, meta: dict[str, Any], transcript: str, analysis: dict[str, Any], job_dir: Path, must_include: list[str] | None = None) -> dict[str, Any]:
    prompt = build_japanese_repair_prompt(url, kind, meta, transcript, analysis, must_include)
    try:
        response = ollama_generate(prompt, job_dir, "japanese_repair", temperature=0.05)
        repaired = parse_json_object(response)
    except Exception as exc:
        (job_dir / "japanese_repair_error.log").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"日本語修復に失敗しました。汎用フォールバック動画は生成しません。原因: {exc}")
    return repaired


def source_points_for_fallback(text: str) -> list[str]:
    text = clean_extracted_text(text or "")
    parts = [p.strip() for p in re.split(r"(?<=[。.!?！？])\s+|\n+", text) if p.strip()]
    scored: list[tuple[int, str]] = []
    for i, part in enumerate(parts):
        if len(part) < 18:
            continue
        score = max(0, 1000 - i)
        if re.search(r"\d|\\$|ドル|円|万|%|RPM|時間|分|日|月|年|再生|収益|Claude|HyperFrames|Google|AI|YouTube", part, flags=re.I):
            score += 500
        if japanese_chars(part) >= 12:
            score += 300
        scored.append((score, part))
    picked = []
    seen = set()
    for _, part in sorted(scored, reverse=True):
        key = part[:80]
        if key in seen:
            continue
        seen.add(key)
        picked.append(part)
        if len(picked) >= 12:
            break
    if len(picked) < 6:
        for part in parts:
            key = part[:80]
            if key not in seen and len(part) >= 18:
                picked.append(part)
                seen.add(key)
            if len(picked) >= 12:
                break
    return picked[:12]


def japanese_fallback_line(point: str, index: int, title: str) -> str:
    point = re.sub(r"https?://\S+", "参照URL", point or "").strip()
    if japanese_chars(point) >= 18:
        return point[:120]
    raise RuntimeError("元資料に基づく日本語フォールバック行を作れませんでした。汎用文で動画化しないため停止します。")


def japanese_extract_fallback(meta: dict[str, Any], transcript: str) -> dict[str, Any]:
    source_title = str(meta.get("title") or "参照動画").strip()
    title = source_title if japanese_chars(source_title) >= 4 else "参照資料の具体要点"
    numbers = re.findall(r"\$?\d[\d,.]*\s?(?:ドル|円|views?|再生|K|万|%|RPM|users?|month|months?)?", transcript, flags=re.I)[:10]
    rough = source_points_for_fallback(transcript or str(meta.get("description") or ""))
    if len(rough) < 8:
        raise RuntimeError("元資料から十分な具体行を抽出できませんでした。汎用フォールバック動画は生成しません。")
    scenes = []
    for i, basis in enumerate(rough[:12]):
        narration = japanese_fallback_line(basis, i, title)
        scenes.append({
            "index": i,
            "narration": narration,
            "image_prompt": "Japanese vertical explainer, clean data cards, bright studio",
            "duration": 8,
        })
    if len(scenes) < 10:
        raise RuntimeError("元資料ベースのシーン数が不足しています。汎用文で水増ししないため停止します。")
    core = scenes[0]["narration"] if scenes else f"{title}の要点を日本語で整理します。"
    return {
        "reference_analysis": {
            "title": f"{title} 要点解説",
            "core_claim": core,
            "evidence_numbers": numbers,
            "workflow_steps": [s["narration"] for s in scenes[:4]],
            "tools_or_methods": [],
            "risks": [],
            "why_it_went_viral": [],
        },
        "scene_plan": {"title": f"{title} 要点解説", "target_duration": 96, "scenes": [{"index": s["index"], "role": "summary", "message": s["narration"], "source_basis": "transcript"} for s in scenes]},
        "script": {"title": f"{title} 要点解説", "scenes": scenes},
        "qa": {"concrete_facts_used": numbers, "omitted_topics": [], "faithfulness_note": "Japanese fallback from transcript"},
    }


def normalize_reference_analysis(analysis: dict[str, Any], meta: dict[str, Any], transcript: str) -> dict[str, Any]:
    reference = analysis.get("reference_analysis")
    if not isinstance(reference, dict):
        reference = {}
    script = analysis.get("script")
    if not isinstance(script, dict):
        script = {}
    title = str(script.get("title") or reference.get("title") or meta.get("title") or "参照動画の要点解説").strip()
    evidence_text = json.dumps(reference.get("evidence_numbers") or [], ensure_ascii=False)
    # Keep the generated title aligned with extracted evidence. Local LLMs can
    # compress "$3,300 / 約50万円" into an inaccurate "月30万円" headline.
    if "3,300" in evidence_text and "50万円" in evidence_text and "月30万円" in title:
        title = title.replace("月30万円", "月50万円")
    scenes = script.get("scenes")
    if not isinstance(scenes, list):
        scenes = []
    cleaned = []
    for i, scene in enumerate(scenes[:12]):
        if not isinstance(scene, dict):
            continue
        narration = str(scene.get("narration") or "").strip()
        if not narration:
            continue
        cleaned.append({
            "index": len(cleaned),
            "narration": narration,
            "image_prompt": str(scene.get("image_prompt") or "clean Japanese vertical explainer, data cards, 9:16").strip()[:180],
            "duration": int(scene.get("duration") or 10),
        })
    if len(cleaned) < 6:
        points = [p.strip() for p in re.split(r"(?<=[.!?。])\\s+|\\n+", transcript) if len(p.strip()) > 25][:12]
        cleaned = [{
            "index": i,
            "narration": point[:95],
            "image_prompt": "clean Japanese vertical explainer, data cards, 9:16",
            "duration": 10,
        } for i, point in enumerate(points)]
    analysis["reference_analysis"] = reference
    analysis["script"] = {"title": title[:70], "scenes": cleaned[:12]}
    plan = analysis.get("scene_plan") if isinstance(analysis.get("scene_plan"), dict) else {}
    plan["title"] = plan.get("title") or title
    plan["target_duration"] = sum(int(s.get("duration") or 10) for s in cleaned[:12])
    if not isinstance(plan.get("scenes"), list) or not plan.get("scenes"):
        plan["scenes"] = [
            {"index": s["index"], "role": "faithful_summary", "message": s["narration"], "source_basis": "transcript"}
            for s in cleaned[:12]
        ]
    analysis["scene_plan"] = plan
    analysis.setdefault("qa", {})
    return analysis


def write_openmontage_artifacts(job_dir: Path, analysis: dict[str, Any]) -> None:
    """Persist OpenMontage-style intermediate artifacts for auditability."""
    (job_dir / "reference_analysis.json").write_text(
        json.dumps(analysis.get("reference_analysis") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "scene_plan.json").write_text(
        json.dumps(analysis.get("scene_plan") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "script.json").write_text(
        json.dumps(analysis.get("script") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "qa.json").write_text(
        json.dumps(analysis.get("qa") or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def enqueue_kurage(
    job_id: str,
    url: str,
    kind: str,
    analysis: dict[str, Any],
    vtuber_mode: bool,
    video_style: str,
    *,
    source: str = "kmontage",
    source_name: str = "Kurage Montage",
) -> str:
    script = analysis.get("script") or {}
    reference = analysis.get("reference_analysis") or {}
    title = str(script.get("title") or reference.get("title") or "参照動画の要点解説").strip()
    payload = {
        "title": title,
        "script": script,
        "source_url": url,
        "source_title": title,
        "source_name": source_name,
        "source_platform": kind,
        "source": source,
        "vtuber_mode": vtuber_mode,
        "video_style": video_style,
    }
    res = requests.post(f"{KURAGE_API}/generate_from_script", json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    kurage_job_id = data.get("job_id")
    if not kurage_job_id:
        raise RuntimeError(f"Kurage did not return job_id: {data}")
    save_job(job_id, kurage_job_id=kurage_job_id, kurage_url=f"https://kurage.exbridge.jp/kuragev.php?id={kurage_job_id}")
    return kurage_job_id


def refresh_from_kurage(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("quality_error") or str(job.get("error") or "").startswith("元資料に忠実な具体台本ではなく"):
        return job
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
        try:
            report = {
                "job_id": job.get("id"),
                "kurage_job_id": kurage_job_id,
                "video_url": updates["video_url"],
                "status": "done",
                "has_script": bool(status.get("script")),
                "scene_count": len((status.get("script") or {}).get("scenes") or []),
                "completed_at": now(),
            }
            job_dir = JOBS_DIR / str(job.get("id"))
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "render_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    elif status.get("status") == "error":
        current_progress = int(job.get("progress") or 0)
        kurage_progress = int(status.get("progress") or 0)
        failed_progress = max(current_progress, 55 + kurage_progress // 3 if kurage_progress else current_progress)
        updates.update({
            "status": "error",
            "progress": min(failed_progress, 99),
            "failed_at_progress": min(failed_progress, 99),
            "error": status.get("error") or "Kurage generation failed",
        })
    else:
        updates.update({"status": "generating", "progress": 55 + int(status.get("progress") or 0) // 3})
    return save_job(job["id"], **updates)


def process_job(job_id: str) -> None:
    job = load_job(job_id) or {}
    url = job.get("url") or ""
    mode = str(job.get("mode") or "summary")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        kind = url_kind(url)
        save_job(job_id, status="analyzing", progress=10, kind=kind, error=None)
        if kind == "x":
            meta = fetch_x_metadata(url, job_dir)
            transcript = captions_from_metadata(meta)
            if len(transcript) < 80 and not meta.get("has_video_media"):
                meta, transcript = fetch_x_article_text(url, job_dir)
            elif len(transcript) < 80 and str(meta.get("description") or "").strip():
                transcript = str(meta.get("description") or "").strip()
        elif is_video_kind(kind):
            meta = fetch_reference_metadata(url, kind, job_dir)
            transcript = captions_from_metadata(meta)
        else:
            meta, transcript = fetch_document_source(url, kind, job_dir)
        save_job(job_id, progress=25, source_title=meta.get("title"), source_uploader=meta.get("uploader") or meta.get("channel"), transcript_preview=transcript[:500])

        video_path = None
        should_transcribe_x_video = (
            kind == "x"
            and bool(meta.get("has_video_media"))
            and float(meta.get("duration") or 0) >= 20
        )
        if is_video_kind(kind) and (kind != "x" or meta.get("has_video_media")) and (len(transcript) < 80 or should_transcribe_x_video):
            pre_video_text = transcript
            save_job(job_id, status="downloading", progress=30)
            video_path = download_reference_video(url, job_dir, kind)
            if video_path:
                save_job(job_id, reference_video=str(video_path), reference_duration=media_duration(video_path))
                save_job(job_id, status="transcribing", progress=38)
                video_transcript = transcribe_video(video_path, job_dir)
                if video_transcript:
                    transcript = clean_extracted_text("\n\n".join(p for p in [pre_video_text, video_transcript] if p))
        save_job(job_id, transcript_preview=transcript[:1000])

        if not transcript and not (meta.get("description") or meta.get("title")):
            raise RuntimeError("入力URLの内容を解析できませんでした。認証済みブラウザ取得などの取得経路が必要です。")

        if mode == "news_opinions":
            save_job(job_id, status="researching", progress=40)
            opinions = collect_news_opinions(url, meta, transcript, job_dir)
            save_job(
                job_id,
                opinion_research=opinions,
                opinion_count=len(opinions.get("opinion_points") or []),
                opinion_sources={
                    "yahoo_comments": len(((opinions.get("sources") or {}).get("yahoo_comments") or [])),
                    "x_replies": len(((opinions.get("sources") or {}).get("x_replies") or [])),
                    "web": len(((opinions.get("sources") or {}).get("web") or [])),
                    "youtube": len(((opinions.get("sources") or {}).get("youtube") or [])),
                    "x": len(((opinions.get("sources") or {}).get("x") or [])),
                },
                opinion_errors=opinions.get("errors") or [],
            )
            save_job(job_id, status="planning", progress=48)
            analysis = analyze_news_opinions(url, kind, meta, transcript, opinions, job_dir)
        else:
            save_job(job_id, status="planning", progress=45)
            analysis = analyze_reference(url, kind, meta, transcript, job_dir)
        reference = analysis.get("reference_analysis") or {}
        scene_plan = analysis.get("scene_plan") or {}
        script = analysis.get("script") or {}
        summary = reference.get("core_claim") or "参照動画の要点を忠実に整理しています。"
        save_job(
            job_id,
            analysis=analysis,
            reference_analysis=reference,
            scene_plan=scene_plan,
            script=script,
            title=script.get("title") or scene_plan.get("title") or reference.get("title"),
            summary=summary,
            script_outline=[s.get("message") or s.get("role") for s in (scene_plan.get("scenes") or []) if isinstance(s, dict)],
        )

        save_job(job_id, status="generating", progress=55)
        if mode == "news_opinions":
            source = "kmontage_news"
            source_name = "Kurage Montage News"
            default_style = "ai_avatar_news_explainer"
        else:
            source = "kmontage"
            source_name = "Kurage Montage"
            default_style = "ai_avatar_explainer"
        kurage_job_id = enqueue_kurage(
            job_id,
            url,
            kind,
            analysis,
            bool(job.get("vtuber_mode", True)),
            str(job.get("video_style") or default_style),
            source=source,
            source_name=source_name,
        )
        save_job(job_id, kurage_job_id=kurage_job_id, status="generating", progress=60)

        deadline = time.time() + 3600
        while time.time() < deadline:
            latest = refresh_from_kurage(load_job(job_id) or {"id": job_id})
            if latest.get("status") in {"done", "error"}:
                return
            time.sleep(15)
        raise RuntimeError("Kurage video generation timed out")
    except Exception as exc:
        current = load_job(job_id) or {}
        failed_progress = int(current.get("progress") or 0)
        save_job(
            job_id,
            status="error",
            error=str(exc),
            progress=min(max(failed_progress, 1), 99),
            failed_at_progress=min(max(failed_progress, 1), 99),
        )


@app.get("/")
def index():
    return {
        "ok": True,
        "service": "kmontage",
        "message": "Kurage Montage API. Public UI is served by kurage/kmontage.php.",
        "health": "/api/health",
        "jobs": "/api/jobs",
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "kmontage",
        "time": now(),
        "kurage_api": KURAGE_API,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "modes": ["summary", "news_opinions"],
        "kagentreach_news_opinion_script": str(KAGENTREACH_NEWS_OPINION_SCRIPT),
    }


@app.post("/api/jobs")
def create_job(req: CreateJobRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="http/https URL を入力してください")
    mode = req.mode.strip() or "summary"
    if mode not in {"summary", "news_opinions"}:
        raise HTTPException(status_code=400, detail="unsupported mode")
    with CREATE_JOB_LOCK:
        active = find_active_job_for_url(url, mode)
        if active:
            return {
                "ok": True,
                "job_id": active["id"],
                "duplicate": True,
                "status": active.get("status"),
                "progress": active.get("progress", 0),
                "message": "同じURLの生成がすでに進行中です。既存の生成状況を表示します。",
            }
        job_id = uuid.uuid4().hex[:16]
        save_job(job_id, id=job_id, url=url, normalized_url=normalize_source_url(url), mode=mode, status="queued", progress=0, vtuber_mode=req.vtuber_mode, video_style=req.video_style, created_at=now())
        thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
        thread.start()
        return {"ok": True, "job_id": job_id, "duplicate": False}


@app.post("/api/jobs/{job_id}/regenerate")
def regenerate_job(job_id: str, req: CreateJobRequest):
    current = load_job(job_id)
    if not current:
        raise HTTPException(status_code=404, detail="job not found")
    if is_active_job(current):
        return {
            "ok": True,
            "job_id": job_id,
            "duplicate": True,
            "regenerated": False,
            "status": current.get("status"),
            "progress": current.get("progress", 0),
            "message": "この動画は生成中です。完了またはエラーになるまで再生成は開始しません。",
        }
    url = req.url.strip() or str(current.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="http/https URL を入力してください")
    mode = req.mode.strip() or str(current.get("mode") or "summary")
    if mode not in {"summary", "news_opinions"}:
        raise HTTPException(status_code=400, detail="unsupported mode")
    old_kurage_job_id = current.get("kurage_job_id")
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    replace_job(job_id, {
        "id": job_id,
        "url": url,
        "normalized_url": normalize_source_url(url),
        "mode": mode,
        "status": "queued",
        "progress": 0,
        "vtuber_mode": req.vtuber_mode,
        "video_style": req.video_style,
        "created_at": current.get("created_at") or now(),
        "regenerated_at": now(),
        "previous_kurage_job_id": old_kurage_job_id,
    })
    thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id, "regenerated": True}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return refresh_from_kurage(job)


@app.get("/api/jobs")
def list_jobs(limit: int = 20, mode: str = ""):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
            if mode and str(job.get("mode") or "summary") != mode:
                continue
            job = normalize_job_progress(job)
            if job.get("kurage_job_id") and job.get("status") not in {"done", "error"}:
                job = refresh_from_kurage(job)
            job = normalize_job_progress(job)
            job["_sort_timestamp"] = job_sort_timestamp(job, p.stat().st_mtime)
            jobs.append(job)
        except Exception:
            pass
    jobs.sort(key=lambda item: float(item.get("_sort_timestamp") or 0), reverse=True)
    for job in jobs:
        job.pop("_sort_timestamp", None)
    jobs = jobs[:limit]
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
