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
    title = meta.get("title") or "参照動画"
    description = meta.get("description") or ""
    uploader = meta.get("uploader") or meta.get("channel") or ""
    duration = meta.get("duration") or ""
    transcript_excerpt = transcript[:14000]
    if len(transcript) > 18000:
        transcript_excerpt += "\n\n--- transcript tail ---\n" + transcript[-4000:]
    return f"""次の参照動画を分析し、日本語ショート動画に再構成してください。

これは「一般論の解説」ではありません。元動画がバズった理由、収益化ノウハウ、具体的な数字、ツール、手順、注意点を忠実に抽出してください。
元動画にない話は入れないでください。考察を入れる場合は「考察」と明示してください。

URL: {url}
種類: {kind}
タイトル: {title}
投稿者: {uploader}
長さ: {duration}秒
説明文:
{description[:1200]}

文字起こし/字幕:
{transcript_excerpt}

要件:
- 日本語で出力
- 元動画を丸ごと転載するのではなく、要点解説・考察にする
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
- reference_analysis: 元動画の事実、数字、手順、注意点、バズった構造
- scene_plan: どの要点をどの順で見せるか
- script: Kurageがそのまま動画化できる12シーン台本

JSONのみで返してください。
形式:
{{
  "reference_analysis": {{
    "title": "元動画の要点タイトル",
    "core_claim": "元動画が主張している中心命題",
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
    write_openmontage_artifacts(job_dir, analysis)
    return analysis


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
        meta = fetch_metadata(url, job_dir)
        transcript = captions_from_metadata(meta)
        save_job(job_id, progress=25, source_title=meta.get("title"), source_uploader=meta.get("uploader") or meta.get("channel"), transcript_preview=transcript[:500])

        video_path = None
        if len(transcript) < 80:
            save_job(job_id, status="downloading", progress=30)
            video_path = download_reference_video(url, job_dir, kind)
            if video_path:
                save_job(job_id, reference_video=str(video_path), reference_duration=media_duration(video_path))
                save_job(job_id, status="transcribing", progress=38)
                transcript = transcribe_video(video_path, job_dir) or transcript
        save_job(job_id, transcript_preview=transcript[:1000])

        if not transcript and not (meta.get("description") or meta.get("title")):
            raise RuntimeError("動画内容を解析できませんでした。認証済みブラウザ録画などの取得経路が必要です。")

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
