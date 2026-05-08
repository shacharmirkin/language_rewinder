# Changelog

## 2026-05-08

- Added fallback model flow from `gemini-2.5-flash` to `gemini-2.5-flash-lite` on `503 UNAVAILABLE`.
- Added exponential backoff retries on fallback with jitter.
- Added "Show all" mode to request all configured decades in one prompt and render as a table.
- Moved app configuration to `config.yaml`.
