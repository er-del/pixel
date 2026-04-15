# inference

This folder provides generation logic for CLI and web usage.

## What It Handles

- Loading model weights from a checkpoint path
- Correcting checkpoint vocab metadata when it disagrees with checkpoint weight shapes
- Sampling with temperature and top-p
- Incremental token streaming for the web endpoint

## Key Files

- `generator.py`: `PixelGenerator` class used by `infer.py` and `web/app.py`

## Related Commands

```bash
python infer.py
python infer.py --prompt "Explain RoPE." --max-tokens 128
python infer.py --model checkpoints/pixel_100m/latest.pt --temperature 0 --top-p 1.0
```
