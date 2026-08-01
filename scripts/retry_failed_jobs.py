#!/usr/bin/env python3
"""Retry failed kmontage jobs serially, reusing existing Kurage work when possible."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
KURAGE_JOBS_DIR = Path("/home/kojima/work/kurage/storage/jobs")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--api", default="http://127.0.0.1:18305")
    value.add_argument("--kurage-api", default="http://127.0.0.1:18303")
    value.add_argument("--job-id", action="append", default=[])
    value.add_argument("--poll-seconds", type=int, default=15)
    value.add_argument("--job-timeout", type=int, default=21600)
    value.add_argument("--max-load-1m", type=float, default=4.0)
    value.add_argument("--min-available-memory-gb", type=float, default=12.0)
    value.add_argument("--capacity-poll-seconds", type=int, default=30)
    value.add_argument("--inter-job-cooldown", type=int, default=60)
    value.add_argument("--report", type=Path, default=ROOT / "storage" / "retry-failed-jobs-report.json")
    return value


def available_memory_gb(meminfo: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def wait_for_capacity(max_load: float, min_memory_gb: float, poll_seconds: int) -> None:
    """Do not begin another expensive job while the shared host is busy."""
    while True:
        load_1m = os.getloadavg()[0]
        memory_gb = available_memory_gb()
        if load_1m <= max_load and memory_gb >= min_memory_gb:
            print(f"[capacity] ready: load1={load_1m:.2f} mem_available={memory_gb:.1f}GiB", flush=True)
            return
        print(
            f"[capacity] waiting: load1={load_1m:.2f}/{max_load:.2f} "
            f"mem_available={memory_gb:.1f}/{min_memory_gb:.1f}GiB",
            flush=True,
        )
        time.sleep(max(5, poll_seconds))


def fetch_jobs(api: str) -> list[dict]:
    response = requests.get(f"{api}/api/jobs", params={"limit": 10000}, timeout=60)
    response.raise_for_status()
    return response.json().get("jobs") or []


def priority(job: dict) -> tuple[int, str]:
    kurage_job_id = str(job.get("kurage_job_id") or "")
    if not kurage_job_id:
        return 3, str(job.get("created_at") or "")
    status = requests.get(f"http://127.0.0.1:18303/status/{kurage_job_id}", timeout=20).json()
    scene_count = len(((status.get("script") or {}).get("scenes") or []))
    asset_count = len(list((KURAGE_JOBS_DIR / kurage_job_id / "assets").glob("scene_*.png")))
    if scene_count and asset_count >= scene_count:
        return 0, str(job.get("created_at") or "")
    if asset_count:
        return 1, str(job.get("created_at") or "")
    return 2, str(job.get("created_at") or "")


def wait_for_terminal(api: str, job_id: str, poll_seconds: int, timeout: int) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = requests.get(f"{api}/api/jobs/{job_id}", timeout=30)
        response.raise_for_status()
        job = response.json()
        marker = (job.get("status"), job.get("progress"), job.get("kurage_progress"))
        if marker != last:
            print(f"[{job_id}] status={marker[0]} progress={marker[1]} kurage={marker[2]}", flush=True)
            last = marker
        if job.get("status") in {"done", "error"}:
            return job
        time.sleep(max(2, poll_seconds))
    raise TimeoutError(f"job {job_id} did not finish within {timeout} seconds")


def retry(api: str, job: dict) -> dict:
    job_id = str(job["id"])
    if job.get("kurage_job_id"):
        response = requests.post(f"{api}/api/jobs/{job_id}/retry-render", timeout=90)
    else:
        payload = {
            "url": job.get("url") or "",
            "vtuber_mode": bool(job.get("vtuber_mode", True)),
            "video_style": job.get("video_style") or "ai_avatar_explainer",
            "mode": job.get("mode") or "summary",
            "editor_mode": job.get("editor_mode") or "normal",
        }
        response = requests.post(f"{api}/api/jobs/{job_id}/regenerate", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parser().parse_args()
    selected_ids = set(args.job_id)
    jobs = [job for job in fetch_jobs(args.api) if job.get("status") == "error"]
    if selected_ids:
        jobs = [job for job in jobs if job.get("id") in selected_ids]
    jobs.sort(key=priority)
    report = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "requested": len(jobs), "results": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    for index, job in enumerate(jobs, 1):
        wait_for_capacity(args.max_load_1m, args.min_available_memory_gb, args.capacity_poll_seconds)
        job_id = str(job["id"])
        print(f"[{index}/{len(jobs)}] retrying {job_id}: {job.get('title') or job.get('url')}", flush=True)
        result = {"job_id": job_id, "title": job.get("title"), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            result["enqueue"] = retry(args.api, job)
            final = wait_for_terminal(args.api, job_id, args.poll_seconds, args.job_timeout)
            result.update({
                "status": final.get("status"),
                "error": final.get("error"),
                "kurage_job_id": final.get("kurage_job_id"),
                "video_url": final.get("video_url") or final.get("kurage_url"),
            })
        except Exception as exc:
            result.update({"status": "retry_error", "error": str(exc)})
        result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report["results"].append(result)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if index < len(jobs) and args.inter_job_cooldown > 0:
            print(f"[cooldown] waiting {args.inter_job_cooldown}s before the next job", flush=True)
            time.sleep(args.inter_job_cooldown)

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(item.get("status") == "done" for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
