"""ensure_model() download-atomicity tests — no real network access.

urllib.request.urlretrieve is monkeypatched so these run offline; the cache
directory is redirected into tmp_path so nothing touches ~/.cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from combat_vision.pose import mediapipe_backend
from combat_vision.pose.mediapipe_backend import ensure_model


@pytest.fixture(autouse=True)
def _cache_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mediapipe_backend, "_model_cache_dir", lambda: tmp_path)


def test_successful_download_lands_at_the_final_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"model bytes")

    monkeypatch.setattr(mediapipe_backend.urllib.request, "urlretrieve", fake_urlretrieve)

    path = ensure_model("lite")

    assert path.exists()
    assert path.read_bytes() == b"model bytes"
    assert path.name == "pose_landmarker_lite.task"
    assert not path.with_suffix(path.suffix + ".part").exists()


def test_failed_download_leaves_no_file_at_the_final_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A network failure mid-download must not leave a corrupt file behind.

    Without the temp-file-then-rename fix, the only freshness check
    (``path.exists()``) would treat this truncated file as a valid,
    already-downloaded model on every future run.
    """

    def failing_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"only partial by")  # simulate a truncated transfer
        raise ConnectionError("connection reset mid-download")

    monkeypatch.setattr(mediapipe_backend.urllib.request, "urlretrieve", failing_urlretrieve)

    with pytest.raises(ConnectionError):
        ensure_model("lite")

    final_path = tmp_path / "pose_landmarker_lite.task"
    assert not final_path.exists()
    assert not final_path.with_suffix(final_path.suffix + ".part").exists()  # cleaned up too


def test_already_cached_model_skips_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cached = tmp_path / "pose_landmarker_lite.task"
    cached.write_bytes(b"already here")

    def unexpected_urlretrieve(url: str, filename: str) -> None:
        raise AssertionError("should not attempt a download when the model is already cached")

    monkeypatch.setattr(mediapipe_backend.urllib.request, "urlretrieve", unexpected_urlretrieve)

    path = ensure_model("lite")
    assert path == cached
    assert path.read_bytes() == b"already here"


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model variant"):
        ensure_model("giant")
