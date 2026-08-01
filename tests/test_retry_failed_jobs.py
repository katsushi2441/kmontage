from pathlib import Path

from scripts.retry_failed_jobs import available_memory_gb, gpu_pressure


def test_available_memory_gb_reads_memavailable(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 65536000 kB\nMemAvailable: 16777216 kB\n", encoding="ascii")

    assert available_memory_gb(meminfo) == 16.0


def test_gpu_pressure_returns_zero_when_monitoring_is_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("scripts.retry_failed_jobs.subprocess.run", unavailable)

    assert gpu_pressure() == (0.0, 0.0)


def test_gpu_pressure_reads_highest_device_values(monkeypatch):
    class Result:
        stdout = "12, 4096\n7, 6144\n"

    monkeypatch.setattr("scripts.retry_failed_jobs.subprocess.run", lambda *_args, **_kwargs: Result())

    assert gpu_pressure() == (12.0, 6144.0)
