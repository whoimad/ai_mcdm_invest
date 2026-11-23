"""The LLM client the real runs use: a local model served by Ollama.

Run settings (decision 6.4): temperature 0, a fixed seed, a 32k context and
JSON output constrained to the schema. The model's digest goes into
`model_version`, so the cache and the results say exactly which weights answered.
"""

import json
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
CONTEXT_TOKENS = 32768
SEED = 42
TIMEOUT_SECONDS = 600


class OllamaClient:
    """Answers a prompt with JSON from `model` on the Ollama server at `host`."""

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
        self.host = host.rstrip("/")
        self.model = model
        models = self._call("/api/tags")["models"]
        digests = {entry["name"]: entry["digest"] for entry in models}
        if model not in digests:
            raise LookupError(f"Ollama has no model {model}; run `ollama pull {model}`")
        self._model_version = f"{model}@sha256:{digests[model]}"

    @property
    def model_version(self) -> str:
        return self._model_version

    def complete(self, prompt: str, schema: dict) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_ctx": CONTEXT_TOKENS},
        }
        return self._call("/api/generate", body)["response"]

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
