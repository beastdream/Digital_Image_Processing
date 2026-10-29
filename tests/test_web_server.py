import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
from server import _parse_byte_range  # noqa: E402


def test_byte_ranges_for_browser_video_playback() -> None:
    assert _parse_byte_range(None, 100) is None
    assert _parse_byte_range("bytes=0-", 100) == (0, 99)
    assert _parse_byte_range("bytes=10-19", 100) == (10, 19)
    assert _parse_byte_range("bytes=-10", 100) == (90, 99)
    assert _parse_byte_range("bytes=0-500", 100) == (0, 99)
    assert _parse_byte_range("bytes=200-", 100) == "invalid"
    assert _parse_byte_range("items=0-1", 100) is None
