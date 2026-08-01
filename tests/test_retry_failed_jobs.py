from pathlib import Path

from scripts.retry_failed_jobs import available_memory_gb, gpu_utilization_percent


def test_available_memory_gb_reads_memavailable(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 65536000 kB\nMemAvailable: 16777216 kB\n", encoding="ascii")

    assert available_memory_gb(meminfo) == 16.0


def test_gpu_utilization_returns_zero_when_monitoring_is_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("scripts.retry_failed_jobs.subprocess.run", unavailable)

    assert gpu_utilization_percent() == 0.0
