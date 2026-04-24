# web

This folder contains the local browser interface for Balchand AI.

## Start The Web App

```bash
python web/app.py
```

Then open `http://127.0.0.1:8000`.

## API Endpoints

- `GET /api/health`: device and checkpoint visibility
- `GET /api/models`: preset names, local checkpoint list, and latest checkpoint metadata
- `GET /api/ui-modes`: available frontend modes
- `POST /api/generate`: full generation response plus resolved checkpoint/config details
- `POST /api/generate/stream`: streamed text output

## Runtime Rules

- The browser keeps the preset selector, but the preset only applies when no checkpoint is active.
- When a checkpoint exists, Balchand AI rebuilds the model from the checkpoint's saved metadata.
- If checkpoint metadata is missing or the tokenizer vocab does not match, the API returns a clear error instead of guessing.

## Common API Errors

- `400 Unknown Balchand AI preset`: the requested `size` is not one of `100m`, `1b`, `3b`, `7b`.
- `400 Tokenizer file was not found`: tokenizer assets are missing; run `python setup.py` or `python train.py`.
- `500 Generation failed due to a backend initialization/runtime error`: check model checkpoint and tokenizer setup, then retry.

## Key Files

- `app.py`: FastAPI app and endpoint contract
- `templates/index.html`: editable UI template with preset controls and active checkpoint status
