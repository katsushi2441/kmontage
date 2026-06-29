#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job


REDIS_URL = "redis://127.0.0.1:6379/0"


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.strip()).strip("-").lower()


def queue_name(ollama_url: str, queue_class: str) -> str:
    host = ollama_url.replace("http://", "").replace("https://", "").split("/", 1)[0].split(":", 1)[0]
    return f"ollama-{slug(host)}-{queue_class}"


def job_result(job: Job) -> Any:
    try:
        return job.return_value()
    except TypeError:
        return job.result


def main() -> int:
    parser = argparse.ArgumentParser(description="Enqueue a kmontage Ollama generation through rqdb4ai queues.")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--ollama-url", default="http://192.168.0.14:11434")
    parser.add_argument("--model", default="gemma4:12b-it-qat")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--queue-class", default="web")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--source", default="web_online")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    result_file = Path(args.result_file)
    redis = Redis.from_url(REDIS_URL)
    qname = queue_name(args.ollama_url, args.queue_class)
    queue = Queue(qname, connection=redis)
    job = queue.enqueue(
        "kmontage_jobs.ollama_generate_job",
        prompt=prompt,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        request_timeout=args.timeout,
        source=args.source,
        meta={
            "project": "kmontage",
            "app": "kmontage",
            "kind": "ollama",
            "resource": "ollama",
            "resource_key": f"ollama:{args.ollama_url}:{args.model}",
            "ollama_host": args.ollama_url.replace("http://", "").replace("https://", "").split("/", 1)[0].split(":", 1)[0],
            "ollama_endpoint": args.ollama_url,
            "ollama_model": args.model,
            "source": args.source,
            "queue_class": args.queue_class,
            "priority_class": "interactive" if args.queue_class == "web" else "background",
            "model": args.model,
        },
        job_timeout=args.timeout + 60,
        result_ttl=86400,
        failure_ttl=604800,
    )

    deadline = time.time() + args.timeout + 45
    while time.time() < deadline:
        job.refresh()
        status = job.get_status(refresh=False)
        if status == "finished":
            result = job_result(job)
            if not isinstance(result, dict):
                raise RuntimeError(f"unexpected rqdb4ai result: {result!r}")
            result_file.write_text(json.dumps({"rq_job_id": job.id, **result}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"ok": True, "rq_job_id": job.id, "queue": qname}, ensure_ascii=False))
            return 0
        if status in {"failed", "stopped", "canceled"}:
            raise RuntimeError(f"rqdb4ai Ollama job {job.id} failed status={status} exc={job.exc_info}")
        time.sleep(2)

    raise RuntimeError(f"rqdb4ai Ollama job timed out job_id={job.id} queue={qname}")


if __name__ == "__main__":
    raise SystemExit(main())
