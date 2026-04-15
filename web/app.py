"""
web/app.py

FastAPI application for browser-based PIXEL inference. This module mirrors the
CLI inference resolution path so the web UI uses the same checkpoint, tokenizer,
and vocab compatibility rules as `infer.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel, Field

from configs.registry import get_preset, list_presets
from core.checkpoint import (
    CheckpointInspection,
    CheckpointManager,
    resolve_inference_vocab_size,
)
from core.runtime import RuntimeManager
from core.types import GenerationRequest
from inference.generator import PixelGenerator
from tokenizer.manager import ensure_tokenizer
from training.bootstrap import ensure_bootstrap_corpus


app = FastAPI(title="PIXEL")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))
RUNTIME = RuntimeManager(PROJECT_ROOT)
UI_MODES = ["chat", "completion", "summarize", "code"]
LOGGER = logging.getLogger("pixel.web")


class GeneratePayload(BaseModel):
    """Request body for text generation."""

    prompt: str = Field(default="Tell me about PIXEL.")
    size: str = Field(default="100m")
    model: str | None = Field(default=None)
    max_tokens: int = Field(default=128)
    temperature: float = Field(default=0.8)
    top_p: float = Field(default=0.95)
    mode: str = Field(default="chat")


def _build_generator(size: str, model_path: str | None) -> PixelGenerator:
    """Create a generator for one request."""
    model_config, _ = get_preset(size)
    checkpoint = model_path or _latest_checkpoint()
    checkpoint_info = _inspect_checkpoint(checkpoint)
    data_path = str(ensure_bootstrap_corpus(PROJECT_ROOT / "data" / "bootstrap" / "demo_corpus.txt"))
    checkpoint_vocab = checkpoint_info.model_config.vocab_size if checkpoint_info is not None else None
    tokenizer_vocab = resolve_inference_vocab_size(checkpoint_info, model_config)
    if checkpoint_vocab is not None and checkpoint_vocab != tokenizer_vocab:
        LOGGER.warning(
            "Corrected checkpoint vocab metadata from %s to %s based on checkpoint weights for %s.",
            checkpoint_vocab,
            tokenizer_vocab,
            model_config.name,
        )
    tokenizer = ensure_tokenizer(
        model_prefix=str(PROJECT_ROOT / "tokenizer" / "pixel_tokenizer"),
        data_paths=[data_path],
        vocab_size=tokenizer_vocab,
    )
    if checkpoint_info is None:
        model_config.vocab_size = tokenizer.vocab_size
    return PixelGenerator(
        model_config,
        tokenizer,
        checkpoint_path=checkpoint,
        checkpoint_info=checkpoint_info,
    )


def _latest_checkpoint() -> str | None:
    """Find the newest PIXEL checkpoint file, if any exist."""
    candidates = sorted((PROJECT_ROOT / "checkpoints").glob("*/latest.pt"))
    if not candidates:
        return None
    return str(candidates[-1])


def _inspect_checkpoint(checkpoint: str | None) -> CheckpointInspection | None:
    """Inspect one checkpoint file or directory when available."""
    if checkpoint is None:
        return None
    target = Path(checkpoint)
    manager = CheckpointManager(target.parent if target.suffix == ".pt" else target, create=False)
    return manager.inspect(path=checkpoint, device="cpu")


def _to_http_exception(exc: Exception) -> HTTPException:
    """Map backend failures to concise API errors."""
    if isinstance(exc, KeyError):
        message = str(exc).strip("'")
        return HTTPException(status_code=400, detail=message)
    if isinstance(exc, (FileNotFoundError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    LOGGER.exception("Unhandled generation failure")
    return HTTPException(
        status_code=500,
        detail=(
            "Generation failed due to a backend initialization/runtime error. "
            "Check tokenizer and checkpoint setup."
        ),
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the PIXEL browser UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"modes": UI_MODES, "sizes": list_presets()},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Return an empty favicon response to avoid noisy browser 404 logs."""
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, object]:
    """Return runtime health details."""
    return RUNTIME.health_payload()


@app.get("/api/models")
def models() -> dict[str, object]:
    """List local checkpoints and preset names."""
    latest = _latest_checkpoint()
    latest_info = _inspect_checkpoint(latest)
    return {
        "presets": list_presets(),
        "checkpoints": RUNTIME.available_checkpoints(),
        "latest": latest,
        "latest_details": latest_info.to_dict() if latest_info is not None else None,
    }


@app.get("/api/ui-modes")
def ui_modes() -> dict[str, list[str]]:
    """Return the supported browser UI modes."""
    return {"modes": UI_MODES}


@app.post("/api/generate")
def generate(payload: GeneratePayload) -> dict[str, object]:
    """Generate text and return the full response."""
    try:
        generator = _build_generator(payload.size, payload.model)
        result = generator.generate(
            GenerationRequest(
                prompt=payload.prompt,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                mode=payload.mode,
            )
        )
    except Exception as exc:
        raise _to_http_exception(exc) from exc
    return asdict(result) | generator.describe()


@app.post("/api/generate/stream")
def generate_stream(payload: GeneratePayload) -> StreamingResponse:
    """Stream generated text chunks over a simple text response."""
    try:
        generator = _build_generator(payload.size, payload.model)
    except Exception as exc:
        raise _to_http_exception(exc) from exc

    def stream():
        try:
            for chunk in generator.stream(
                GenerationRequest(
                    prompt=payload.prompt,
                    max_tokens=payload.max_tokens,
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                    mode=payload.mode,
                )
            ):
                yield chunk
        except Exception:  # pragma: no cover - stream execution path
            LOGGER.exception("Streaming generation failed")
            yield "\n[PIXEL ERROR] Streaming generation failed. Check server logs.\n"

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
