# tests

This folder contains targeted tests for the Balchand AI rewrite.

## Run Tests

```bash
pytest -q
```

## Coverage Areas

- Preset config correctness
- Tokenizer setup and round-trip behavior
- Model forward shape and weight tying
- Checkpoint save/load
- Data normalization and token dataset windows
- Web API health and generation responses
