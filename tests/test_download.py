"""Failed downloads must never leave a file at the destination, and an
interrupted download must resume from its .part file."""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import torchxrayvision as xrv


def fake_response(status=200, headers=None, chunks=(), content=b"", error=None):
    """A stand-in for a streaming requests.Response. `error` is raised after `chunks`."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.headers = headers or {}
    response.content = content

    def iter_content(**kwargs):
        yield from chunks
        if error:
            raise error

    response.iter_content.side_effect = iter_content
    return response


class TestWeightDownload(unittest.TestCase):
    @contextmanager
    def mock_weights(self, cache, responses):
        """Yield (requests.get mock, call) where call() runs get_weights against `responses`."""
        weights = {"download-test": {"weights_url": "https://example.invalid/weights.pt"}}
        with patch.dict(xrv.models.model_urls, weights), patch.object(
            xrv.utils.requests, "get", side_effect=responses
        ) as get:
            yield get, lambda: xrv.models.get_weights("download-test", cache_dir=cache)

    def test_http_error_leaves_nothing_and_is_retried(self):
        with tempfile.TemporaryDirectory() as cache:
            bad = fake_response(status=500, content=b"error page")
            bad.raise_for_status.side_effect = OSError("HTTP failure")
            good = fake_response(headers={"content-length": "8"}, chunks=[b"complete"])
            target = Path(cache) / "weights.pt"

            with self.mock_weights(cache, [bad, good]) as (get, call):
                with self.assertRaises(OSError):
                    call()
                self.assertEqual(list(Path(cache).iterdir()), [])

                self.assertEqual(Path(call()), target)
                self.assertEqual(target.read_bytes(), b"complete")
                self.assertEqual(list(Path(cache).iterdir()), [target])

    def test_interrupted_download_resumes_from_part_file(self):
        with tempfile.TemporaryDirectory() as cache:
            cut = fake_response(
                headers={"content-length": "14"}, chunks=[b"partial-"], error=OSError("interrupted")
            )
            rest = fake_response(
                status=206, headers={"content-length": "6"}, chunks=[b"result"]
            )
            target = Path(cache) / "weights.pt"
            part = Path(cache) / "weights.pt.part"

            with self.mock_weights(cache, [cut, rest]) as (get, call):
                with self.assertRaises(OSError):
                    call()
                self.assertFalse(target.exists())
                self.assertEqual(part.read_bytes(), b"partial-")

                self.assertEqual(Path(call()), target)
                self.assertEqual(get.call_args.kwargs["headers"], {"Range": "bytes=8-"})
                self.assertEqual(target.read_bytes(), b"partial-result")
                self.assertEqual(list(Path(cache).iterdir()), [target])

    def test_server_ignoring_range_restarts_download(self):
        with tempfile.TemporaryDirectory() as cache:
            (Path(cache) / "weights.pt.part").write_bytes(b"stale")
            full = fake_response(headers={"content-length": "8"}, chunks=[b"complete"])
            target = Path(cache) / "weights.pt"

            with self.mock_weights(cache, [full]) as (_, call):
                call()
                self.assertEqual(target.read_bytes(), b"complete")
                self.assertEqual(list(Path(cache).iterdir()), [target])

    def test_unsatisfiable_range_discards_part_and_restarts(self):
        with tempfile.TemporaryDirectory() as cache:
            (Path(cache) / "weights.pt.part").write_bytes(b"complete")
            unsatisfiable = fake_response(status=416)
            full = fake_response(headers={"content-length": "8"}, chunks=[b"complete"])
            target = Path(cache) / "weights.pt"

            with self.mock_weights(cache, [unsatisfiable, full]) as (get, call):
                call()
                self.assertEqual(get.call_count, 2)
                self.assertEqual(get.call_args_list[0].kwargs["headers"], {"Range": "bytes=8-"})
                self.assertEqual(get.call_args_list[1].kwargs["headers"], {})
                self.assertEqual(target.read_bytes(), b"complete")
                self.assertEqual(list(Path(cache).iterdir()), [target])
