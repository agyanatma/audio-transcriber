# audio-transcriber

A dependency-light command-line tool that transcribes audio and video files
using OpenRouter's `openai/gpt-4o-mini-transcribe` model. No Python packages
to install — just the standard library, `ffmpeg`, and an API key.

## Features

- **Zero Python dependencies** — runs with a stock Python 3.10+ install.
- **Any format ffmpeg can read** — mp3, wav, m4a, mp4, mov, and more.
- **Handles long recordings** — files of 10 minutes or longer are
  automatically split into sequential chunks (5 minutes by default) so
  requests stay within OpenRouter's timeout.
- **Resilient** — automatically retries on rate limits and transient
  network/server errors.
- **Simple config** — reads your API key from a local `.env` file, no
  shell exports required.

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10 or newer |
| `ffmpeg` / `ffprobe` | Used to inspect and normalize input audio |
| OpenRouter API key | Get one at [openrouter.ai](https://openrouter.ai) |

Install ffmpeg on macOS with Homebrew:

```sh
brew install ffmpeg
```

## Setup

1. Clone the repo and enter it:

   ```sh
   git clone https://github.com/agyanatma/audio-transcriber.git
   cd audio-transcriber
   ```

2. Add your API key:

   ```sh
   cp .env.example .env
   # Edit .env and replace the placeholder with your OpenRouter API key.
   ```

That's it — no dependency installation needed to use the `./transcribe` script.

## Usage

Run it directly from the repo:

```sh
./transcribe recording.mp3
./transcribe meeting.m4a --output meeting.txt
./transcribe interview.mp4 --language en --output interview.txt
```

By default the transcript prints to stdout, so it can be redirected:

```sh
./transcribe recording.wav > recording.txt
```

Progress messages go to stderr, so redirecting stdout doesn't hide them.

Prefer installing it as a regular command? Use a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install .
audio-transcriber recording.mp3
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output PATH` | stdout | Write the transcript to a file instead of stdout |
| `-l`, `--language CODE` | auto-detect | ISO-639-1 hint, e.g. `en`, `id`, `ja` |
| `--chunk-minutes N` | `5` | Chunk length used for files of 10 minutes or longer |
| `--temperature N` | `0` | Sampling temperature, `0`–`1` |
| `--timeout SECONDS` | `120` | Per-request HTTP timeout |
| `--retries N` | `3` | Retries for rate limits and transient failures |
| `--force` | off | Overwrite an existing output file |
| `-q`, `--quiet` | off | Hide progress messages |

Run `./transcribe --help` for the full CLI reference.

## How it works

1. `ffprobe` measures the source duration.
2. `ffmpeg` normalizes each chunk to mono 16 kHz MP3.
3. The CLI sends base64-encoded audio to OpenRouter's
   `POST /api/v1/audio/transcriptions` endpoint.
4. Chunk transcripts are stitched back together in order, separated by a
   blank line.

OpenRouter enforces an upstream timeout of ~60 seconds per request for large
audio payloads. Five-minute chunks keep individual requests small; lower
`--chunk-minutes` if a particularly dense file still times out.

## Configuration precedence

The CLI reads `.env` from the current directory automatically. An existing
shell environment variable of the same name always takes precedence over the
value in `.env`.

## Testing

```sh
python3 -m pip install -e .
python3 -m unittest discover tests
```
