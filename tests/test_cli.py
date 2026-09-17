from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from audio_transcriber.cli import (
    LONG_AUDIO_THRESHOLD_SECONDS,
    TRANSCRIPTION_MODEL,
    build_chunks,
    load_env_file,
    select_model,
    transcribe_file,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class CliTests(unittest.TestCase):
    def test_model_boundary(self) -> None:
        self.assertEqual(
            select_model(LONG_AUDIO_THRESHOLD_SECONDS - 0.001),
            TRANSCRIPTION_MODEL,
        )
        self.assertEqual(select_model(LONG_AUDIO_THRESHOLD_SECONDS), TRANSCRIPTION_MODEL)
        self.assertEqual(
            select_model(LONG_AUDIO_THRESHOLD_SECONDS + 1),
            TRANSCRIPTION_MODEL,
        )

    def test_build_chunks_preserves_duration(self) -> None:
        chunks = build_chunks(1250, 300)

        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[-1].start_seconds, 1200)
        self.assertEqual(chunks[-1].duration_seconds, 50)
        self.assertAlmostEqual(sum(chunk.duration_seconds for chunk in chunks), 1250)

    def test_load_env_file(self) -> None:
        original = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                env_file = Path(directory) / ".env"
                env_file.write_text(
                    '# comment\nOPENROUTER_API_KEY="from-file"\n',
                    encoding="utf-8",
                )
                load_env_file(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-file")

                os.environ["OPENROUTER_API_KEY"] = "existing"
                load_env_file(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "existing")
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)
            if original is not None:
                os.environ["OPENROUTER_API_KEY"] = original

    def test_transcription_payload(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            captured["authorization"] = request.headers["Authorization"]
            return FakeResponse({"text": "hello", "usage": {"cost": 0.01}})

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "chunk.mp3"
            audio.write_bytes(b"audio")
            result = transcribe_file(
                audio,
                api_key="secret",
                model=TRANSCRIPTION_MODEL,
                language="en",
                temperature=0,
                timeout=12,
                max_retries=0,
                opener=opener,
            )

        self.assertEqual(result["text"], "hello")
        self.assertEqual(captured["payload"]["model"], TRANSCRIPTION_MODEL)
        self.assertEqual(captured["payload"]["input_audio"]["format"], "mp3")
        self.assertEqual(captured["payload"]["language"], "en")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["authorization"], "Bearer secret")

    def test_retries_temporary_http_error(self) -> None:
        calls = []
        sleeps = []

        def opener(_request, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise HTTPError(
                    url="https://example.test",
                    code=429,
                    msg="rate limited",
                    hdrs={"Retry-After": "0"},
                    fp=io.BytesIO(b'{"error":{"message":"rate limited"}}'),
                )
            return FakeResponse({"text": "recovered"})

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "chunk.mp3"
            audio.write_bytes(b"audio")
            result = transcribe_file(
                audio,
                api_key="secret",
                model=TRANSCRIPTION_MODEL,
                language=None,
                temperature=0,
                timeout=10,
                max_retries=1,
                opener=opener,
                sleep=sleeps.append,
            )

        self.assertEqual(result["text"], "recovered")
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0])


if __name__ == "__main__":
    unittest.main()
