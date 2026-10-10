import json
import os
from pathlib import Path

import httpx
import pytest

from agentqa.evaluators import (
    DeepSeekConfigurationError,
    DeepSeekJudgeProvider,
    DeepSeekJudgeSettings,
    JudgeRequest,
    load_judge_environment,
)


def _request() -> JudgeRequest:
    return JudgeRequest(
        prompt='{"instruction":"Return JSON","evidence":{"answer":"ok"}}',
        response_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
        },
        attempt=1,
    )


def _settings(**updates: object) -> DeepSeekJudgeSettings:
    values: dict[str, object] = {
        "api_key": "test-secret-key",
        "model": "deepseek-test",
        "retry_backoff_seconds": 0.0,
    }
    values.update(updates)
    return DeepSeekJudgeSettings(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_deepseek_provider_requests_json_and_records_safe_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "model": "deepseek-resolved",
                "system_fingerprint": "fingerprint-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "PASS",
                                    "score": 1.0,
                                    "rationale": "Correct.",
                                    "violations": [],
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                    "unexpected": "ignored",
                },
            },
        )

    provider = DeepSeekJudgeProvider(
        _settings(),
        transport=httpx.MockTransport(handler),
    )
    response = await provider.judge(_request())

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer test-secret-key"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0
    assert "schema" in body["messages"][0]["content"]
    assert response.metadata["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }
    assert response.metadata["request_id"] == "request-1"
    assert "test-secret-key" not in repr(provider.parameters)
    assert "test-secret-key" not in repr(_settings())


@pytest.mark.asyncio
async def test_deepseek_provider_retries_transient_status_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"verdict":"NEEDS_REVIEW","score":null,'
                                '"rationale":"Ambiguous.","violations":[]}'
                            )
                        },
                    }
                ]
            },
        )

    provider = DeepSeekJudgeProvider(
        _settings(),
        transport=httpx.MockTransport(handler),
    )

    await provider.judge(_request())

    assert calls == 2


def test_deepseek_configuration_rejects_non_official_key_destination() -> None:
    with pytest.raises(DeepSeekConfigurationError, match="official"):
        _settings(base_url="https://example.test/v1")


def test_private_judge_environment_loads_only_known_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AGENTQA_JUDGE_API_KEY",
        "AGENTQA_JUDGE_BASE_URL",
        "AGENTQA_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / ".env.local"
    source.write_text(
        "AGENTQA_JUDGE_API_KEY='test-secret-key'\n"
        "AGENTQA_JUDGE_BASE_URL=https://api.deepseek.com\n"
        "AGENTQA_JUDGE_MODEL=deepseek-test\n",
        encoding="utf-8",
    )
    source.chmod(0o600)

    assert load_judge_environment(source) is True
    provider = DeepSeekJudgeProvider.from_environment()

    assert provider.name == "deepseek"
    assert provider.model == "deepseek-test"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are unavailable")
def test_judge_environment_rejects_broad_permissions(tmp_path: Path) -> None:
    source = tmp_path / ".env.local"
    source.write_text("AGENTQA_JUDGE_MODEL=deepseek-test\n", encoding="utf-8")
    source.chmod(0o644)

    with pytest.raises(DeepSeekConfigurationError, match="group or others"):
        load_judge_environment(source)
