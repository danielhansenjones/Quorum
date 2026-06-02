from __future__ import annotations

import os
import socket
import time
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _vllm_url() -> str | None:
    return os.environ.get("VLLM_URL")


def _tcp_reachable(url: str, timeout: float = 1.5) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def vllm_client():
    url = _vllm_url()
    if not url:
        pytest.skip(
            "VLLM_URL not set; bring up vllm with `docker compose --profile gpu up -d vllm`"
        )
    if not _tcp_reachable(url):
        pytest.skip(f"vllm not reachable at {url}")
    from openai import OpenAI

    return OpenAI(base_url=url.rstrip("/") + "/v1", api_key="not-used")


def test_models_endpoint(vllm_client) -> None:
    models = vllm_client.models.list()
    assert len(models.data) >= 1


def test_simple_completion(vllm_client) -> None:
    resp = vllm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[{"role": "user", "content": "Reply with only the word OK."}],
        max_tokens=8,
        temperature=0.0,
    )
    assert resp.choices
    assert resp.choices[0].message.content


def test_streaming_completion(vllm_client) -> None:
    stream = vllm_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[{"role": "user", "content": "Count to three."}],
        max_tokens=32,
        temperature=0.0,
        stream=True,
    )
    chunks = list(stream)
    assert len(chunks) > 0


def test_concurrent_batching_beats_sequential(vllm_client) -> None:
    # Continuous batching gate from Phase 2b. 8 concurrent should be meaningfully
    # faster than 8 serial; we set a generous 1.5x floor to absorb cold-cache
    # noise.
    import concurrent.futures

    def call_once() -> None:
        vllm_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            messages=[{"role": "user", "content": "Reply with one short sentence."}],
            max_tokens=24,
            temperature=0.0,
        )

    start_seq = time.monotonic()
    for _ in range(8):
        call_once()
    seq = time.monotonic() - start_seq

    start_par = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: call_once(), range(8)))
    par = time.monotonic() - start_par

    assert par * 1.5 < seq, (
        f"concurrent batch {par:.2f}s did not beat sequential {seq:.2f}s; "
        "continuous batching is not engaging"
    )
