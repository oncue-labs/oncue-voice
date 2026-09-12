import json
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oncue_voice.conversation.events import ConversationEvent
from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.conversation.split_pipeline_runtime import (
    SplitPipelineRuntime,
    SplitPipelineRuntimeOptions,
)
from oncue_voice.providers.split_pipeline.factory import ProviderFactory
from oncue_voice.providers.split_pipeline.models import ProviderSettings


PCM_SAMPLE_RATE_HZ = 24_000
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNEL_COUNT = 1


@dataclass(frozen=True)
class SplitPipelineEvaluationRequest:
    run_id: str
    variant_id: str
    combination_key: str
    input_text: str
    policy: DialoguePolicy
    provider_settings: ProviderSettings
    input_audio: bytes
    artifact_root: Path


@dataclass(frozen=True)
class SplitPipelineEvaluationResult:
    artifact_directory: Path
    succeeded: bool


class SplitPipelineEvaluationService:
    def __init__(self, provider_factory: ProviderFactory) -> None:
        self._provider_factory = provider_factory

    async def run(
        self,
        request: SplitPipelineEvaluationRequest,
    ) -> SplitPipelineEvaluationResult:
        providers = self._provider_factory.create(request.provider_settings)
        conversation_events: list[ConversationEvent] = []
        runtime = SplitPipelineRuntime(
            providers,
            SplitPipelineRuntimeOptions(event_sink=conversation_events.append),
        )
        started_at = time.perf_counter()
        response_audio = b"".join(
            [
                chunk
                async for chunk in runtime.run(
                    request.run_id,
                    request.policy,
                    self._input_audio(request.input_audio),
                )
            ]
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)

        artifact_directory = request.artifact_root / request.run_id / request.variant_id
        artifact_directory.mkdir(parents=True, exist_ok=True)
        self._write_artifacts(
            request,
            artifact_directory,
            conversation_events,
            response_audio,
            elapsed_ms,
        )
        return SplitPipelineEvaluationResult(
            artifact_directory=artifact_directory,
            succeeded=True,
        )

    @staticmethod
    async def _input_audio(audio: bytes) -> AsyncIterator[bytes]:
        yield audio

    @classmethod
    def _write_artifacts(
        cls,
        request: SplitPipelineEvaluationRequest,
        artifact_directory: Path,
        conversation_events: list[ConversationEvent],
        response_audio: bytes,
        elapsed_ms: int,
    ) -> None:
        cls._write_json(
            artifact_directory / "run.json",
            {
                "runId": request.run_id,
                "variantId": request.variant_id,
                "combinationKey": request.combination_key,
                "provider": request.provider_settings.provider,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "succeeded": True,
                "elapsedMs": elapsed_ms,
                "audioBytes": len(response_audio),
            },
        )
        cls._write_json(
            artifact_directory / "policy-snapshot.json",
            request.policy.model_dump(mode="json", by_alias=True),
        )
        cls._write_json(
            artifact_directory / "provider-config.json",
            request.provider_settings.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        )
        cls._write_json(
            artifact_directory / "input.json",
            {
                "combinationKey": request.combination_key,
                "inputText": request.input_text,
                "source": "synthetic",
            },
        )
        cls._write_json(
            artifact_directory / "transcript.json",
            {
                "events": [
                    event.model_dump(mode="json", exclude_none=False)
                    for event in conversation_events
                ]
            },
        )
        cls._write_wave(artifact_directory / "response.wav", response_audio)
        (artifact_directory / "evaluation.md").write_text(
            cls._evaluation_template(request),
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_wave(path: Path, audio: bytes) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(PCM_CHANNEL_COUNT)
            output.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
            output.setframerate(PCM_SAMPLE_RATE_HZ)
            output.writeframes(audio)

    @staticmethod
    def _evaluation_template(request: SplitPipelineEvaluationRequest) -> str:
        return f"""# Voice evaluation

- Run: `{request.run_id}`
- Variant: `{request.variant_id}`
- Combination: `{request.combination_key}`

## Manual scores (1–5)

- Voice quality:
- Persona consistency:
- Scenario goal:
- Safety: pass / violation

## Comments

"""
