import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import backend.main as main


class KurageReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.jobs_dir = Path(self.tempdir.name)
        self.jobs_patch = patch.object(main, "JOBS_DIR", self.jobs_dir)
        self.jobs_patch.start()

    def tearDown(self) -> None:
        self.jobs_patch.stop()
        self.tempdir.cleanup()

    def write_job(self, job_id: str, **values: object) -> None:
        payload = {"id": job_id, **values}
        (self.jobs_dir / f"{job_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @patch("backend.main.requests.get")
    def test_stale_timeout_error_is_reconciled_to_done(self, get: Mock) -> None:
        self.write_job(
            "stale",
            status="error",
            error="Kurage video generation timed out",
            progress=72,
            kurage_job_id="kurage-done",
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "status": "done",
            "progress": 100,
            "title": "completed video",
            "script": {"scenes": [{"index": 0}]},
        }
        get.return_value = response

        refreshed = main.refresh_from_kurage(main.load_job("stale") or {})

        self.assertEqual(refreshed["status"], "done")
        self.assertEqual(refreshed["progress"], 100)
        self.assertIsNone(refreshed["error"])
        self.assertEqual(
            refreshed["video_url"],
            "https://kurage.exbridge.jp/kuragev.php?id=kurage-done",
        )

    def test_restart_preserves_job_already_enqueued_to_kurage(self) -> None:
        self.write_job(
            "rendering",
            status="generating",
            progress=70,
            kurage_job_id="kurage-active",
        )

        main.mark_interrupted_jobs_on_startup()

        preserved = main.load_job("rendering") or {}
        self.assertEqual(preserved["status"], "generating")
        self.assertIsNone(preserved["error"])
        self.assertEqual(preserved["kurage_job_id"], "kurage-active")
        self.assertTrue(preserved.get("monitoring_deferred_at"))

    def test_restart_marks_pre_render_work_as_interrupted(self) -> None:
        self.write_job("planning", status="planning", progress=48)

        main.mark_interrupted_jobs_on_startup()

        interrupted = main.load_job("planning") or {}
        self.assertEqual(interrupted["status"], "error")
        self.assertIn("worker thread was interrupted", interrupted["error"])

    @patch("backend.main.requests.get")
    def test_unchanged_kurage_error_does_not_touch_updated_at(self, get: Mock) -> None:
        self.write_job(
            "failed",
            status="error",
            error="ERNIE image generation failed",
            progress=66,
            failed_at_progress=66,
            updated_at="2026-07-01 00:00:00",
            kurage_job_id="kurage-failed",
            kurage_status="error",
            kurage_progress=33,
            kurage_title="failed video",
            kurage_script=None,
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "status": "error",
            "progress": 33,
            "title": "failed video",
            "script": None,
            "error": "ERNIE image generation failed",
        }
        get.return_value = response

        refreshed = main.refresh_from_kurage(main.load_job("failed") or {})

        self.assertEqual(refreshed["updated_at"], "2026-07-01 00:00:00")

    @patch("backend.main.requests.get")
    def test_job_list_does_not_poll_terminal_infrastructure_errors(self, get: Mock) -> None:
        self.write_job(
            "failed",
            status="error",
            error="Voicebox TTS failed",
            kurage_job_id="kurage-failed",
        )

        response = main.list_jobs(limit=20)

        self.assertEqual(response["jobs"][0]["status"], "error")
        get.assert_not_called()

    def test_concurrent_job_updates_keep_every_field(self) -> None:
        self.write_job("concurrent", status="generating")

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda index: main.save_job("concurrent", **{f"field_{index}": index}), range(20)))

        saved = main.load_job("concurrent") or {}
        for index in range(20):
            self.assertEqual(saved[f"field_{index}"], index)


if __name__ == "__main__":
    unittest.main()
