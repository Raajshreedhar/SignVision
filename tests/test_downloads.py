"""src.download_data: skip, cache and validation paths (no network).

kagglehub.dataset_download and urllib.request.urlopen are monkeypatched, and
config.RAW_DIR / config.SOURCES_FILE point into tmp_path.
"""

import io
import json
import urllib.error
import urllib.request
import zipfile

import pytest

from src import config
from src import download_data as dd


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    sources = tmp_path / "data" / "sources.json"
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SOURCES_FILE", sources)
    return raw, sources


class FakeResponse:
    """Minimal urlopen() response: headers, read(n), context manager."""

    def __init__(self, body, content_length="auto"):
        self._buf = io.BytesIO(body)
        if content_length is None:
            self.headers = {}
        else:
            size = len(body) if content_length == "auto" else content_length
            self.headers = {"Content-Length": str(size)}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_zip(members=None):
    members = members or {"images/train/a.jpg": b"jpegbytes", "labels/train/a.txt": b"0 0.5 0.5 0.2 0.2\n"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def patch_urlopen(monkeypatch, response_or_exc):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def patch_kaggle(monkeypatch, result_or_exc):
    import kagglehub

    def fake_download(handle, *args, **kwargs):
        if isinstance(result_or_exc, Exception):
            raise result_or_exc
        return str(result_or_exc)

    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)


HK = dd.REGISTRY["hand_keypoints"]


def test_registry_has_six_sources_with_two_required():
    assert list(dd.REGISTRY) == [
        "grassknoted", "debashishsau", "ayuraj", "danrasband", "hand_keypoints", "hagrid30k",
    ]
    assert {n for n, s in dd.REGISTRY.items() if s.required} == {"grassknoted", "hand_keypoints"}
    assert dd.REGISTRY["grassknoted"].class_root == "asl_alphabet_train/asl_alphabet_train"
    assert dd.REGISTRY["grassknoted"].test_glob == "asl_alphabet_test/asl_alphabet_test/*_test.jpg"
    assert dd.REGISTRY["hagrid30k"].role == "det"


def test_kaggle_failure_is_skipped_with_credential_hint(monkeypatch):
    patch_kaggle(monkeypatch, PermissionError("403 Forbidden"))
    result = dd.fetch(dd.REGISTRY["ayuraj"])
    assert result.status == "skipped"
    assert result.path is None
    assert "~/.kaggle/kaggle.json" in result.message
    assert "KAGGLE_USERNAME" in result.message and "KAGGLE_KEY" in result.message
    assert "PermissionError" in result.message and "403 Forbidden" in result.message


def test_kaggle_success_is_ok(monkeypatch, tmp_path):
    patch_kaggle(monkeypatch, tmp_path / "cache" / "ayuraj")
    result = dd.fetch(dd.REGISTRY["ayuraj"])
    assert result.status == "ok"
    assert result.path == tmp_path / "cache" / "ayuraj"


def test_url_zip_already_complete_short_circuits(monkeypatch, tmp_data, capsys):
    raw, sources = tmp_data
    dest = raw / "hand_keypoints"
    dest.mkdir(parents=True)
    (dest / ".complete").write_text("done\n")
    calls = patch_urlopen(monkeypatch, AssertionError("urlopen must not be called"))

    assert dd.main(["--sources", "hand_keypoints"]) == 0
    assert calls == []
    data = json.loads(sources.read_text())
    assert data["hand_keypoints"]["path"] == str(dest)
    assert "fetched_at" in data["hand_keypoints"]
    assert "already present" in capsys.readouterr().err


def test_content_length_mismatch_deletes_partial(monkeypatch, tmp_data):
    raw, _ = tmp_data
    body = make_zip()
    patch_urlopen(monkeypatch, FakeResponse(body, content_length=len(body) + 100))
    result = dd.fetch(HK)
    assert result.status == "failed"
    assert "size mismatch" in result.message
    assert not (raw / "hand_keypoints.zip.part").exists()
    assert not (raw / "hand_keypoints.zip").exists()
    assert not (raw / "hand_keypoints").exists()


def test_missing_content_length_with_valid_zip_is_ok(monkeypatch, tmp_data):
    raw, _ = tmp_data
    patch_urlopen(monkeypatch, FakeResponse(make_zip(), content_length=None))
    result = dd.fetch(HK)
    assert result.status == "ok", result.message
    dest = raw / "hand_keypoints"
    assert result.path == dest
    assert (dest / ".complete").exists()
    assert (dest / "labels" / "train" / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert not (raw / "hand_keypoints.zip").exists()
    assert not (raw / "hand_keypoints.zip.part").exists()


def test_not_a_zip_fails(monkeypatch, tmp_data):
    raw, _ = tmp_data
    patch_urlopen(monkeypatch, FakeResponse(b"definitely not a zip archive" * 10))
    result = dd.fetch(HK)
    assert result.status == "failed"
    assert not (raw / "hand_keypoints.zip.part").exists()
    assert not (raw / "hand_keypoints").exists()


def test_corrupt_zip_member_fails(monkeypatch, tmp_data):
    raw, _ = tmp_data
    name = "labels/train/a.txt"
    body = bytearray(make_zip({name: b"0 0.5 0.5 0.2 0.2\n"}))
    body[30 + len(name) + 2] ^= 0xFF  # flip one byte of the stored member data -> CRC mismatch
    patch_urlopen(monkeypatch, FakeResponse(bytes(body)))
    result = dd.fetch(HK)
    assert result.status == "failed"
    assert "corrupt" in result.message
    assert not (raw / "hand_keypoints.zip.part").exists()
    assert not (raw / "hand_keypoints").exists()


def test_network_error_fails(monkeypatch, tmp_data):
    patch_urlopen(monkeypatch, urllib.error.URLError("connection reset"))
    result = dd.fetch(HK)
    assert result.status == "failed"
    assert "URLError" in result.message


def test_merge_sources_json_keeps_existing_entries(tmp_data):
    _, sources = tmp_data
    sources.parent.mkdir(parents=True)
    sources.write_text(json.dumps({"grassknoted": {"path": "/cache/g", "fetched_at": "2025-01-01T00:00:00+00:00"}}))
    results = [
        dd.FetchResult("hand_keypoints", "ok", sources.parent / "raw" / "hand_keypoints", "fresh"),
        dd.FetchResult("hagrid30k", "skipped", None, "no credentials"),
    ]
    merged = dd.merge_sources_json(results)
    on_disk = json.loads(sources.read_text())
    assert merged == on_disk
    assert on_disk["grassknoted"]["path"] == "/cache/g"
    assert on_disk["hand_keypoints"]["path"].endswith("hand_keypoints")
    assert "hagrid30k" not in on_disk
    assert [p.name for p in sources.parent.iterdir()] == ["sources.json"]  # no temp file left behind


def test_exit_1_when_required_source_fails(monkeypatch, tmp_data, capsys):
    patch_urlopen(monkeypatch, urllib.error.URLError("offline"))
    assert dd.main(["--sources", "hand_keypoints"]) == 1
    err = capsys.readouterr().err
    assert "hand_keypoints" in err and "failed" in err
    assert "Traceback" not in err


def test_exit_0_when_only_optional_sources_fail(monkeypatch, tmp_data, tmp_path, capsys):
    raw, sources = tmp_data
    (raw / "hand_keypoints").mkdir(parents=True)
    (raw / "hand_keypoints" / ".complete").write_text("done\n")
    calls = {}

    import kagglehub

    def fake_download(handle, *args, **kwargs):
        calls[handle] = True
        if handle == "grassknoted/asl-alphabet":
            return str(tmp_path / "cache" / "grassknoted")
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)
    code = dd.main(["--sources", "grassknoted,hand_keypoints,hagrid30k,ayuraj"])
    assert code == 0
    data = json.loads(sources.read_text())
    assert set(data) == {"grassknoted", "hand_keypoints"}
    err = capsys.readouterr().err
    assert "skipped" in err and "~/.kaggle/kaggle.json" in err
    assert "Traceback" not in err


def test_list_and_bad_sources(capsys):
    assert dd.main(["--list"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 + len(dd.REGISTRY)
    with pytest.raises(SystemExit) as exc:
        dd.main(["--sources", "bogus"])
    assert exc.value.code == 2
