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

import requests
from bs4 import BeautifulSoup
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
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.0.3:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b-it-qat")
YTDLP_BIN = os.environ.get("YTDLP_BIN", "yt-dlp")
YTDLP_COOKIES_FILE = os.environ.get("KMONTAGE_YTDLP_COOKIES_FILE", "")
YTDLP_COOKIES_BROWSER = os.environ.get("KMONTAGE_YTDLP_COOKIES_BROWSER", "")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")
ENABLE_TRANSCRIBE = os.environ.get("KMONTAGE_ENABLE_TRANSCRIBE", "1").lower() not in {"0", "false", "no"}
KURAGEVP_BACKEND_DIR = Path(os.environ.get("KURAGEVP_BACKEND_DIR", "/home/kojima/work/kuragevp/backend"))

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
        "duration": 0,
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
    transcript_excerpt = transcript[:14000]
    if len(transcript) > 18000:
        transcript_excerpt += "\n\n--- transcript tail ---\n" + transcript[-4000:]
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
    return json.loads(text)


def analyze_reference(url: str, kind: str, meta: dict[str, Any], transcript: str, job_dir: Path) -> dict[str, Any]:
    prompt = build_analysis_prompt(url, kind, meta, transcript)
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 8192}}
    res = requests.post(ollama_generate_url(), json=payload, timeout=300)
    res.raise_for_status()
    response = res.json().get("response") or ""
    (job_dir / "analysis_response.txt").write_text(response, encoding="utf-8")
    try:
        analysis = parse_json_object(response)
    except Exception:
        title = meta.get("title") or "参照動画の要点解説"
        base = transcript or meta.get("description") or title
        points = [line.strip() for line in re.split(r"[。\n]", base) if line.strip()][:8]
        scenes = []
        for i, point in enumerate(points[:12]):
            scenes.append({
                "index": i,
                "narration": point[:90],
                "image_prompt": "clean Japanese vertical explainer, data cards, 9:16",
                "duration": 10,
            })
        analysis = {
            "reference_analysis": {
                "title": f"{title} 要点解説",
                "core_claim": base[:200],
                "evidence_numbers": re.findall(r"(?:\$?\d[\d,.]*\s?(?:ドル|円|views?|再生|K|万|%|RPM)?)", base)[:8],
                "workflow_steps": points[:6],
                "tools_or_methods": [],
                "risks": [],
                "why_it_went_viral": [],
            },
            "scene_plan": {"title": f"{title} 要点解説", "target_duration": 120, "scenes": []},
            "script": {"title": f"{title} 要点解説", "scenes": scenes},
            "qa": {"concrete_facts_used": points[:4], "omitted_topics": [], "faithfulness_note": "fallback from transcript"},
        }
    analysis = normalize_reference_analysis(analysis, meta, transcript)
    if needs_japanese_repair(analysis):
        repaired = repair_analysis_to_japanese(url, kind, meta, transcript, analysis, job_dir)
        analysis = normalize_reference_analysis(repaired, meta, transcript)
    if needs_japanese_repair(analysis):
        (job_dir / "japanese_quality_error.json").write_text(json.dumps({
            "reason": "script_is_not_japanese_enough",
            "title": (analysis.get("script") or {}).get("title"),
            "scene_count": len(((analysis.get("script") or {}).get("scenes") or [])),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("日本語ショート台本の生成に失敗しました。英語台本のままKurageへ送信しないため停止しました。")
    write_openmontage_artifacts(job_dir, analysis)
    return analysis


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


def build_japanese_repair_prompt(url: str, kind: str, meta: dict[str, Any], transcript: str, analysis: dict[str, Any]) -> str:
    label = source_label(kind)
    title = meta.get("title") or label
    uploader = meta.get("uploader") or meta.get("channel") or ""
    description = meta.get("description") or ""
    transcript_excerpt = transcript[:14000]
    if len(transcript) > 18000:
        transcript_excerpt += "\n\n--- transcript tail ---\n" + transcript[-4000:]
    previous = json.dumps(analysis, ensure_ascii=False)[:8000]
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


def repair_analysis_to_japanese(url: str, kind: str, meta: dict[str, Any], transcript: str, analysis: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    prompt = build_japanese_repair_prompt(url, kind, meta, transcript, analysis)
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.05, "num_predict": 8192}}
    res = requests.post(ollama_generate_url(), json=payload, timeout=300)
    res.raise_for_status()
    response = res.json().get("response") or ""
    (job_dir / "japanese_repair_response.txt").write_text(response, encoding="utf-8")
    try:
        repaired = parse_json_object(response)
    except Exception as exc:
        (job_dir / "japanese_repair_error.log").write_text(str(exc), encoding="utf-8")
        repaired = japanese_extract_fallback(meta, transcript)
    return repaired


def japanese_extract_fallback(meta: dict[str, Any], transcript: str) -> dict[str, Any]:
    title = str(meta.get("title") or "参照動画").strip()
    numbers = re.findall(r"\$?\d[\d,.]*\s?(?:ドル|円|views?|再生|K|万|%|RPM|users?|month|months?)?", transcript, flags=re.I)[:10]
    rough = [p.strip() for p in re.split(r"(?<=[.!?。])\s+|\n+", transcript) if len(p.strip()) > 35][:12]
    scenes = []
    for i in range(12):
        basis = rough[i] if i < len(rough) else (rough[-1] if rough else title)
        scenes.append({
            "index": i,
            "narration": f"元動画の要点です。{basis[:80]}。この内容を日本語で整理すると、具体的な数字と手順を確認することが重要です。",
            "image_prompt": "Japanese vertical explainer, clean data cards, bright studio",
            "duration": 8,
        })
    return {
        "reference_analysis": {
            "title": f"{title} 要点解説",
            "core_claim": "元動画の主張を日本語で要約できなかったため、抽出字幕をもとに再構成しました。",
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


def enqueue_kurage(job_id: str, url: str, kind: str, analysis: dict[str, Any], vtuber_mode: bool, video_style: str) -> str:
    script = analysis.get("script") or {}
    reference = analysis.get("reference_analysis") or {}
    title = str(script.get("title") or reference.get("title") or "参照動画の要点解説").strip()
    payload = {
        "title": title,
        "script": script,
        "source_url": url,
        "source_title": title,
        "source_name": "Kurage Montage",
        "source_platform": kind,
        "source": "kmontage",
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
        if is_video_kind(kind) and (kind != "x" or meta.get("has_video_media")) and len(transcript) < 80:
            save_job(job_id, status="downloading", progress=30)
            video_path = download_reference_video(url, job_dir, kind)
            if video_path:
                save_job(job_id, reference_video=str(video_path), reference_duration=media_duration(video_path))
                save_job(job_id, status="transcribing", progress=38)
                transcript = transcribe_video(video_path, job_dir) or transcript
        save_job(job_id, transcript_preview=transcript[:1000])

        if not transcript and not (meta.get("description") or meta.get("title")):
            raise RuntimeError("入力URLの内容を解析できませんでした。認証済みブラウザ取得などの取得経路が必要です。")

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
    return {"ok": True, "service": "kmontage", "time": now(), "kurage_api": KURAGE_API, "ollama_url": OLLAMA_URL, "ollama_model": OLLAMA_MODEL}


@app.post("/api/jobs")
def create_job(req: CreateJobRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="http/https URL を入力してください")
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
