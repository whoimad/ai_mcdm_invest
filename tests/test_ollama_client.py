"""Ollama client: the run settings of decision 6.4 go out with every request.

A small local HTTP server stands in for Ollama, so no model is needed.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_mcdm_invest.llm.ollama import OllamaClient

DIGEST = "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"


class FakeOllama(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_GET(self) -> None:
        assert self.path == "/api/tags"
        models = [
            {"name": "other:latest", "digest": "0" * 64},
            {"name": "qwen2.5:7b", "digest": DIGEST},
        ]
        self._reply({"models": models})

    def do_POST(self) -> None:
        assert self.path == "/api/generate"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        self._reply({"response": '{"quote": ""}', "done": True})

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def host() -> Iterator[str]:
    FakeOllama.requests = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_model_version_records_the_model_digest(host: str) -> None:
    client = OllamaClient(host=host)

    assert client.model_version == f"qwen2.5:7b@sha256:{DIGEST}"


def test_every_request_sends_the_run_settings_and_schema(host: str) -> None:
    schema = {"type": "object", "properties": {"quote": {"type": "string"}}}

    reply = OllamaClient(host=host).complete("Read this.", schema)

    assert reply == '{"quote": ""}'
    [request] = FakeOllama.requests
    assert request["model"] == "qwen2.5:7b"
    assert request["prompt"] == "Read this."
    assert request["format"] == schema
    assert request["stream"] is False
    assert request["options"]["temperature"] == 0
    assert request["options"]["num_ctx"] == 32768
    assert isinstance(request["options"]["seed"], int)


def test_a_model_not_pulled_is_an_error(host: str) -> None:
    with pytest.raises(LookupError, match="llama3.1:8b"):
        OllamaClient(host=host, model="llama3.1:8b")
