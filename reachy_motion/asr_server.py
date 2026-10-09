"""``reachy-motion-asr``: a standalone Breeze-ASR-26 (MLX) speech-recognition server for the Taiwanese mode.

OpenAI-compatible ``POST /v1/audio/transcriptions`` (multipart ``file``), ``GET /health``, ``GET /v1/models``, on
Apple Silicon via mlx-audio. Same model and decoding settings as the AIRI NTU-VH2026 lab server (MIT), so Reachy
no longer depends on the AIRI desktop package. ``reachy-motion-node`` shares it on the LAN.

    pip install -e ".[asr]"
    reachy-motion-asr                                   # downloads RayyTien/Breeze-ASR-26-mlx-4bit on first run
    reachy-motion-asr --model-dir /path/to/model --port 8001
"""


import argparse
import io
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("reachy_motion.asr")

MODEL_REPO = "RayyTien/Breeze-ASR-26-mlx-4bit"
MODEL_ID = "breeze-asr-26-mlx"
MAX_SECONDS = 120


def decode_audio(data: bytes) -> np.ndarray:
    """Any container/codec (wav, webm, mp3, m4a) -> 16 kHz mono float32, via PyAV's bundled FFmpeg."""
    import av

    chunks = []
    with av.open(io.BytesIO(data)) as container:
        rs = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for out in rs.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in rs.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    if len(audio) / 16000 > MAX_SECONDS:
        raise ValueError(f"audio longer than {MAX_SECONDS} s")
    return audio


def patch_no_speech_token(model) -> dict:
    """NOTICE: mlx-audio 0.4.3 looks up ``<|nospeech|>``, but this tokenizer spells it ``<|nocaptions|>``.

    The unknown-token fallback equals EOT, so the decoder suppresses its own end of text and never stops.
    Patch the wrapper's ``no_speech`` property in memory (site-packages untouched), as the AIRI lab server does
    (course/ntu-vh2026/asr26/whisper-compat.py). Remove once mlx-audio resolves the HF spelling itself.
    """
    wrapper = model.get_tokenizer(language="zh", task="transcribe")
    vocab = wrapper.hf_tokenizer.get_vocab()
    eot = wrapper.hf_tokenizer.eos_token_id
    token_id = next((int(vocab[s]) for s in ("<|nocaptions|>", "<|nospeech|>") if s in vocab), None)
    if token_id is None or token_id == eot:
        raise RuntimeError("no safe no-speech token in this tokenizer")
    type(wrapper).no_speech = property(lambda self: token_id)
    return {"no_speech_id": token_id, "eot_id": eot}


class Engine:
    """All MLX work runs on ONE worker thread: MLX GPU streams belong to the thread that created them."""

    def __init__(self, model_dir: str | None) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.model_dir = model_dir
        self.model = None
        self.info: dict = {}
        self._cc = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def load(self) -> None:
        self.worker.submit(self._load).result()

    def _load(self) -> None:
        import mlx.core as mx
        from mlx_audio.stt.utils import load

        path = self.model_dir
        if not path:
            from huggingface_hub import snapshot_download

            path = snapshot_download(MODEL_REPO)
        t0 = time.perf_counter()
        self.model = load(str(Path(path).expanduser()), strict=True)
        self.info = {"model_dir": str(path), **patch_no_speech_token(self.model)}
        mx.synchronize()
        self.info["load_s"] = round(time.perf_counter() - t0, 2)
        try:  # Breeze often answers in Simplified characters; Reachy speaks Taiwan Mandarin
            from opencc import OpenCC

            self._cc = OpenCC("s2twp")
        except Exception:  # noqa: BLE001 - optional
            self._cc = None

    def transcribe(self, audio: np.ndarray):
        """Future of ``(text, seconds)``; requests queue on the single MLX worker."""
        return self.worker.submit(self._transcribe, audio)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        import mlx.core as mx

        t0 = time.perf_counter()
        r = self.model.generate(audio, language="zh", task="transcribe", temperature=0.0, return_timestamps=False,
                                word_timestamps=False, condition_on_previous_text=False, verbose=None)
        mx.synchronize()
        dt = time.perf_counter() - t0
        text = r.text.strip()
        return (self._cc.convert(text) if self._cc else text), dt


def create_app(engine: Engine):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(title="reachy-motion-asr")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok" if engine.model else "loading", "model": MODEL_ID, "source": MODEL_REPO, **engine.info}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str = Form(MODEL_ID),  # noqa: B008
                             language: str = Form("zh"), response_format: str = Form("json")):
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "audio upload exceeds 25 MiB")
        try:
            audio = decode_audio(data)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"cannot decode audio: {e}") from None
        if not len(audio):
            raise HTTPException(422, "empty audio")
        import asyncio

        text, dt = await asyncio.wrap_future(engine.transcribe(audio))
        headers = {"X-Inference-Seconds": f"{dt:.3f}", "X-Audio-Seconds": f"{len(audio) / 16000:.3f}"}
        if response_format == "text":
            return PlainTextResponse(text, headers=headers)
        body = {"text": text}
        if response_format == "verbose_json":
            body |= {"duration": len(audio) / 16000, "language": language}
        return JSONResponse(body, headers=headers)

    return app


def main() -> None:
    p = argparse.ArgumentParser(prog="reachy-motion-asr", description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", default=os.environ.get("BREEZE_ASR26_MODEL_DIR"),
                   help=f"local model folder (default: download {MODEL_REPO} from Hugging Face)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    engine = Engine(a.model_dir)
    logger.info("loading %s …", a.model_dir or MODEL_REPO)
    engine.load()
    logger.info("ready: %s", engine.info)
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
