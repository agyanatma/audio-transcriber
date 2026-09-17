# audio-transcriber

A dependency-light CLI for transcribing local audio through OpenRouter.

- All audio uses `openai/gpt-4o-mini-transcribe`.
- Long files are split into sequential five-minute chunks by default.
- Any audio or video format readable by ffmpeg can be used as input.

## Requirements

- Python 3.10 or newer
- `ffmpeg` and `ffprobe`
- An OpenRouter API key

On macOS with Homebrew:

```sh
brew install ffmpeg
```

## Run

Copy the environment template and add your API key:

```sh
cp .env.example .env
# Edit .env and replace the placeholder value.
```

Use the executable directly from this repository:

```sh
./transcribe recording.mp3
./transcribe meeting.m4a --output meeting.txt
./transcribe interview.mp4 --language en --output interview.txt
```

The transcript is printed to stdout unless `--output` is supplied. Progress is
written to stderr, so redirection also works:

```sh
./transcribe recording.wav > recording.txt
```

The CLI automatically reads `.env` from the current directory. A shell
environment variable with the same name takes precedence when one is set.

Install it as a command in a virtual environment if preferred:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install .
audio-transcriber recording.mp3
```

Useful options:

```text
--chunk-minutes 5   Chunk length for files of 10 minutes or longer
--language id       Optional ISO-639-1 language hint
--timeout 120       Per-request HTTP timeout
--retries 3         Retries for temporary API failures
--force             Overwrite an existing output file
--quiet             Hide progress
```

Run `./transcribe --help` for the complete CLI reference.

## How it works

The CLI uses `ffprobe` to measure the source duration, then uses `ffmpeg` to
normalize each request to mono 16 kHz MP3. It calls OpenRouter's dedicated
`POST /api/v1/audio/transcriptions` endpoint with base64-encoded audio. Chunk
transcripts are joined in source order with a blank line between chunks.

OpenRouter documents an upstream provider timeout of 60 seconds for large
audio requests. Five-minute chunks keep requests smaller; lower
`--chunk-minutes` if a difficult file still times out.
