"""Transcribe LifeMate audio with VinAI PhoWhisper.

Examples (run from the repository root):

    # Transcribe the latest request of every public-test sample.
    python3 main/asr.py --split public_test --output artifacts/public_test_asr.jsonl

    # Use a larger model on GPU and include user audio in conversation history.
    python3 main/asr.py --split public_test --model vinai/PhoWhisper-medium \
        --device cuda --include-history

The output is deliberately separate from public_test.jsonl: it adds ASR text
without modifying competition data. An agent can join rows by ``id``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import wave


class PhoWhisperTranscriber:
    """Lazy wrapper around the official Hugging Face PhoWhisper checkpoints."""

    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise SystemExit(
                "Missing ASR dependencies. Install PyTorch for your platform, then run "
                "`pip install -r requirements.txt`."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")

        # Transformers pipeline uses an integer device index: -1 means CPU.
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    def transcribe(self, audio_path: Path) -> str:
        """Return Vietnamese transcription for one .wav file."""
        audio = load_wav(audio_path)
        result = self._pipe(
            audio,
            # Greedy deterministic decoding is required for reproducible
            # competition submissions.
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "do_sample": False,
                "num_beams": 1,
            },
        )
        return str(result["text"]).strip()


def load_wav(path: Path) -> dict[str, Any]:
    """Read uncompressed WAV directly, so the pipeline does not require ffmpeg.

    Competition audio is 16 kHz, mono, signed-16-bit PCM. The additional PCM
    handling keeps this utility useful for other WAV recordings as well.
    """
    try:
        import numpy as np
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            width = source.getsampwidth()
            if source.getcomptype() != "NONE":
                raise ValueError(f"compressed WAV is not supported: {source.getcomptype()}")
            raw = source.readframes(source.getnframes())
    except (wave.Error, OSError) as exc:
        raise ValueError(f"cannot read WAV file {path}: {exc}") from exc

    if channels < 1 or width not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported WAV format: {channels} channel(s), {width * 8}-bit")
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (values[:, 0].astype(np.int32) | (values[:, 1].astype(np.int32) << 8) | (values[:, 2].astype(np.int32) << 16))
        samples = ((samples ^ (1 << 23)) - (1 << 23)).astype(np.float32) / (1 << 23)
    else:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return {"array": samples, "sampling_rate": sample_rate}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def user_history_transcripts(
    item: dict[str, Any], split_dir: Path, transcriber: PhoWhisperTranscriber
) -> list[dict[str, str]]:
    """Transcribe only user turns; assistant text/tool calls are already textual."""
    output: list[dict[str, str]] = []
    for turn in item.get("history", []):
        if turn.get("role") == "user" and turn.get("audio"):
            output.append({
                "audio": turn["audio"],
                "audio_text": transcriber.transcribe(split_dir / turn["audio"]),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe LifeMate audio with PhoWhisper")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository/data root")
    parser.add_argument("--split", choices=("train", "public_test"), default="public_test")
    parser.add_argument("--model", default="vinai/PhoWhisper-small", help="Hugging Face PhoWhisper model ID")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="ASR JSONL destination")
    parser.add_argument("--include-history", action="store_true", help="also transcribe user turns in history")
    parser.add_argument("--limit", type=int, help="number of samples to process (useful for a smoke test)")
    args = parser.parse_args()

    split_dir = args.root / args.split
    input_path = split_dir / f"{args.split}.jsonl"
    if not input_path.is_file():
        raise SystemExit(f"Dataset not found: {input_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    transcriber = PhoWhisperTranscriber(args.model, args.device)

    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for item in read_jsonl(input_path):
            audio_path = split_dir / item["audio"]
            row: dict[str, Any] = {
                "id": item["id"],
                "audio": item["audio"],
                "audio_text": transcriber.transcribe(audio_path),
            }
            if args.include_history:
                row["history_user_asr"] = user_history_transcripts(item, split_dir, transcriber)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            print(f"[{count}] {item['id']}: {row['audio_text']}", flush=True)
            if args.limit is not None and count >= args.limit:
                break

    print(f"Wrote {count} transcript(s) to {args.output}")


if __name__ == "__main__":
    main()
