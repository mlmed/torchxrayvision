"""Download failures must leave a missing weight eligible for a later retry."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torchxrayvision as xrv


class TestWeightDownloadRetry(unittest.TestCase):
    def test_failed_first_download_is_retried(self):
        for failure in ("http", "interrupted"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as cache:
                response = MagicMock()
                response.__enter__.return_value = response
                response.headers = {}
                if failure == "http":
                    response.raise_for_status.side_effect = OSError("HTTP failure")
                else:
                    def interrupted_chunks(**kwargs):
                        yield b"partial"
                        raise OSError("interrupted transfer")

                    response.iter_content.side_effect = interrupted_chunks

                successful = MagicMock()
                successful.__enter__.return_value = successful
                successful.headers = {"content-length": "8"}
                successful.iter_content.return_value = [b"complete"]
                weights = {"download-test": {"weights_url": "https://example.invalid/weights.pt"}}
                target = Path(cache) / "weights.pt"

                with patch.dict(xrv.models.model_urls, weights), patch.object(
                    xrv.utils.requests, "get", side_effect=[response, successful]
                ) as get:
                    with self.assertRaises(OSError):
                        xrv.models.get_weights("download-test", cache_dir=cache)
                    self.assertFalse(target.exists())
                    self.assertEqual(list(Path(cache).iterdir()), [])

                    downloaded = xrv.models.get_weights("download-test", cache_dir=cache)
                    self.assertEqual(Path(downloaded), target)
                    self.assertEqual(target.read_bytes(), b"complete")
                    self.assertEqual(get.call_count, 2)
                    self.assertEqual(list(Path(cache).iterdir()), [target])

                response.__exit__.assert_called_once()
                successful.__exit__.assert_called_once()
