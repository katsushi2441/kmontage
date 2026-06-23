from __future__ import annotations

import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "audio path required"}, ensure_ascii=False))
        return 2
    audio = Path(sys.argv[1])
    if not audio.exists():
        print(json.dumps({"ok": False, "error": f"not found: {audio}"}, ensure_ascii=False))
        return 2
    model_name = sys.argv[2] if len(sys.argv) >= 3 else "small"
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, info = model.transcribe(str(audio), language=None, vad_filter=True)
    text_parts = []
    segment_items = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        text_parts.append(text)
        segment_items.append({"start": seg.start, "end": seg.end, "text": text})
    print(json.dumps({
        "ok": True,
        "language": getattr(info, "language", ""),
        "duration": getattr(info, "duration", 0),
        "text": " ".join(text_parts),
        "segments": segment_items,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
