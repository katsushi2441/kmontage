from pathlib import Path

from scripts.retry_failed_jobs import available_memory_gb


def test_available_memory_gb_reads_memavailable(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 65536000 kB\nMemAvailable: 16777216 kB\n", encoding="ascii")

    assert available_memory_gb(meminfo) == 16.0
