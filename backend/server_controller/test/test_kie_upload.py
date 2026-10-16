"""Kie image uploads use the file header, not the .bin storage name."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from kie_client import (  # noqa: E402
    is_binary_object_url,
    kie_hosted_file_url,
    kie_upload_is_image,
    sniff_image,
    upload_file_bytes,
)


def test_download_url_is_the_hosted_file():
    body = {
        "success": True,
        "code": 200,
        "msg": "File uploaded successfully",
        "data": {
            "fileName": "cf6157ff.bin",
            "downloadUrl": "https://tempfile.redpandaai.co/kieai/12073064/images/cf6157ff.bin",
            "mimeType": "application/octet-stream",
        },
    }
    assert kie_hosted_file_url(body).endswith("cf6157ff.bin")
    assert kie_upload_is_image(body) is False


def test_older_file_url_still_counts():
    body = {"code": 200, "data": {"fileUrl": "https://tempfile.redpandaai.co/a.png"}}
    assert kie_hosted_file_url(body) == "https://tempfile.redpandaai.co/a.png"
    assert kie_upload_is_image(body) is True


def test_file_header_sets_the_image_type():
    png_name, png_mime = sniff_image(b"\x89PNG\r\n\x1a\nrest")
    jpg_name, jpg_mime = sniff_image(b"\xff\xd8\xffrest")
    assert png_name.endswith(".png")
    assert png_mime == "image/png"
    assert jpg_name.endswith(".jpg")
    assert jpg_mime == "image/jpeg"
    assert png_name != jpg_name
    assert sniff_image(b"not an image") is None


def test_bin_storage_url_is_not_an_image_url():
    stored = "https://r2.example/input-media/cf6157ff.bin?X-Amz-Signature=abc"
    assert is_binary_object_url(stored) is True
    assert is_binary_object_url("https://cdn.example/photo.webp") is False


def test_stream_upload_sends_the_sniffed_image():
    response = MagicMock()
    response.json.return_value = {
        "code": 200,
        "success": True,
        "data": {
            "fileName": "abc.jpg",
            "downloadUrl": "https://tempfile.redpandaai.co/images/abc.jpg",
            "mimeType": "image/jpeg",
        },
    }
    with patch("kie_client.requests.post", return_value=response) as post:
        url = upload_file_bytes(b"\xff\xd8\xffhello", "photo.bin", "application/octet-stream")
    assert url.endswith("abc.jpg")
    filename, _data, content_type = post.call_args.kwargs["files"]["file"]
    assert filename.endswith(".jpg")
    assert content_type == "image/jpeg"
    assert post.call_args.kwargs["data"]["fileName"] == filename
