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
    assert list(path.parent.glob("*.part")) == []


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
    assert list(tmp_path.glob("*.part")) == []  # cleaned up too


def test_concurrent_downloads_of_the_same_variant_do_not_collide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two invocations racing on the same variant must not share one temp file.

    Before the fix, both downloads derived the same ``.part`` path from the
    variant name alone; one invocation's rename/cleanup could stomp the
    other's still-in-flight write. Recursing into a second ensure_model()
    call mid-download reproduces that race without real threads: at that
    point ``path.exists()`` is still False for both, so both proceed.
    """
    seen_tmp_names: list[str] = []
    calls = 0

    def racing_urlretrieve(url: str, filename: str) -> None:
        nonlocal calls
        seen_tmp_names.append(filename)
        calls += 1
        if calls == 1:
            ensure_model("lite")  # a second invocation starts mid-download
        Path(filename).write_bytes(b"model bytes")

    monkeypatch.setattr(mediapipe_backend.urllib.request, "urlretrieve", racing_urlretrieve)

    path = ensure_model("lite")

    assert len(seen_tmp_names) == 2
    assert len(set(seen_tmp_names)) == 2  # distinct temp files, never a shared one
    assert path.exists()
    assert path.read_bytes() == b"model bytes"
    assert list(tmp_path.glob("*.part")) == []  # nothing left behind


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
