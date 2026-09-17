from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
TRANSCRIPTION_MODEL = "openai/gpt-4o-mini-transcribe"
LONG_AUDIO_THRESHOLD_SECONDS = 10 * 60
DEFAULT_CHUNK_SECONDS = 5 * 60
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 524, 529}


class TranscriptionError(RuntimeError):
    """A user-facing transcription failure."""


@dataclass(frozen=True)
class Chunk:
    index: int
    total: int
    start_seconds: float
    duration_seconds: float


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding existing variables."""
    if not path.is_file():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TranscriptionError(f"Could not read environment file {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise TranscriptionError(
                f"Invalid entry in {path} at line {line_number}: expected KEY=VALUE."
            )

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise TranscriptionError(
                f"Invalid variable name in {path} at line {line_number}."
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise TranscriptionError(
            f"Required executable '{name}' was not found. Install ffmpeg and try again."
        )
    return path


def get_duration(path: Path, ffprobe: str) -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe could not read the file"
        raise TranscriptionError(f"Could not determine audio duration: {detail}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise TranscriptionError("ffprobe returned an invalid audio duration.") from exc

    if not math.isfinite(duration) or duration <= 0:
        raise TranscriptionError("The input file does not contain usable audio.")
    return duration


def build_chunks(duration_seconds: float, chunk_seconds: float) -> list[Chunk]:
    total = max(1, math.ceil(duration_seconds / chunk_seconds))
    return [
        Chunk(
            index=index,
            total=total,
            start_seconds=index * chunk_seconds,
            duration_seconds=min(chunk_seconds, duration_seconds - index * chunk_seconds),
        )
        for index in range(total)
    ]


def select_model(duration_seconds: float) -> str:
    return TRANSCRIPTION_MODEL


def normalize_chunk(source: Path, destination: Path, chunk: Chunk, ffmpeg: str) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{chunk.start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{chunk.duration_seconds:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "96k",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg could not convert the audio"
        raise TranscriptionError(
            f"Could not prepare chunk {chunk.index + 1}/{chunk.total}: {detail}"
        )
    if not destination.exists() or destination.stat().st_size == 0:
        raise TranscriptionError(
            f"ffmpeg produced an empty chunk at {format_timestamp(chunk.start_seconds)}."
        )


def parse_retry_after(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def extract_error_message(body: bytes) -> str:
    if not body:
        return ""
    try:
        data = json.loads(body)
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or "")
        return str(error)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace").strip()[:500]


def transcribe_file(
    audio_path: Path,
    *,
    api_key: str,
    model: str,
    language: str | None,
    temperature: float,
    timeout: float,
    max_retries: int,
    api_url: str = API_URL,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
        "model": model,
        "input_audio": {"data": audio_data, "format": "mp3"},
        "temperature": temperature,
    }
    if language:
        payload["language"] = language

    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "audio-transcriber CLI",
        },
        method="POST",
    )

    for attempt in range(max_retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                response_body = response.read()
                result = json.loads(response_body)
                if not isinstance(result, dict):
                    raise TranscriptionError("OpenRouter returned an unexpected JSON response.")
                text = result.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise TranscriptionError(
                        "OpenRouter returned a successful response without transcript text."
                    )
                return result
        except HTTPError as exc:
            try:
                body = exc.read()
            finally:
                exc.close()
            message = extract_error_message(body)
            if exc.code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                retry_after = parse_retry_after(exc.headers)
                delay = retry_after if retry_after is not None else min(2**attempt, 15)
                sleep(delay)
                continue
            detail = f": {message}" if message else ""
            raise TranscriptionError(f"OpenRouter request failed ({exc.code}){detail}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt < max_retries:
                sleep(min(2**attempt, 15))
                continue
            reason = getattr(exc, "reason", exc)
            raise TranscriptionError(f"Could not reach OpenRouter: {reason}") from exc
        except json.JSONDecodeError as exc:
            raise TranscriptionError("OpenRouter returned invalid JSON.") from exc

    raise AssertionError("unreachable")


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def write_output(text: str, output: Path | None, force: bool) -> None:
    if output is None:
        print(text)
        return
    if output.exists() and not force:
        raise TranscriptionError(
            f"Output file already exists: {output}. Use --force to overwrite it."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-transcriber",
        description="Transcribe audio through OpenRouter.",
    )
    parser.add_argument("audio_file", type=Path, help="Audio or video file to transcribe")
    parser.add_argument("-o", "--output", type=Path, help="Write transcript to this file")
    parser.add_argument(
        "-l",
        "--language",
        help="Optional ISO-639-1 language code, such as en, id, or ja",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=DEFAULT_CHUNK_SECONDS / 60,
        help="Chunk size for audio of 10 minutes or longer (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Transcription sampling temperature from 0 to 1 (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for each request (default: 120)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for temporary API failures (default: 3)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide progress messages")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.audio_file.is_file():
        raise TranscriptionError(f"Input file does not exist: {args.audio_file}")
    if args.output and args.output.exists() and not args.force:
        raise TranscriptionError(
            f"Output file already exists: {args.output}. Use --force to overwrite it."
        )
    if not 0 < args.chunk_minutes <= 10:
        raise TranscriptionError("--chunk-minutes must be greater than 0 and at most 10.")
    if not 0 <= args.temperature <= 1:
        raise TranscriptionError("--temperature must be between 0 and 1.")
    if args.timeout <= 0:
        raise TranscriptionError("--timeout must be greater than 0.")
    if args.retries < 0:
        raise TranscriptionError("--retries cannot be negative.")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    load_env_file()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise TranscriptionError(
            "Set OPENROUTER_API_KEY in .env or as an environment variable."
        )

    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")
    duration = get_duration(args.audio_file, ffprobe)
    is_long = duration >= LONG_AUDIO_THRESHOLD_SECONDS
    model = select_model(duration)
    chunk_seconds = args.chunk_minutes * 60 if is_long else duration
    chunks = build_chunks(duration, chunk_seconds)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    progress(f"Duration: {format_duration(duration)}")
    progress(f"Model: {model}")
    if is_long:
        progress(f"Chunks: {len(chunks)} × up to {args.chunk_minutes:g} minutes")

    transcript_parts: list[str] = []
    total_cost = 0.0
    with tempfile.TemporaryDirectory(prefix="audio-transcriber-") as temp_dir:
        for chunk in chunks:
            progress(
                f"[{chunk.index + 1}/{chunk.total}] Preparing "
                f"{format_timestamp(chunk.start_seconds)}–"
                f"{format_timestamp(chunk.start_seconds + chunk.duration_seconds)}"
            )
            chunk_path = Path(temp_dir) / f"chunk-{chunk.index + 1:04d}.mp3"
            normalize_chunk(args.audio_file, chunk_path, chunk, ffmpeg)

            progress(f"[{chunk.index + 1}/{chunk.total}] Transcribing")
            result = transcribe_file(
                chunk_path,
                api_key=api_key,
                model=model,
                language=args.language,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.retries,
            )
            transcript_parts.append(result["text"].strip())
            usage = result.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
                total_cost += float(usage["cost"])

    transcript = "\n\n".join(transcript_parts)
    write_output(transcript, args.output, args.force)
    destination = str(args.output) if args.output else "stdout"
    cost_suffix = f"; reported cost ${total_cost:.6f}" if total_cost else ""
    progress(f"Done: {destination}{cost_suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except TranscriptionError as exc:
        parser.exit(1, f"error: {exc}\n")
