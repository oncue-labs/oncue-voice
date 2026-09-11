import json
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from oncue_voice.conversation.models import DialoguePolicy
from oncue_voice.conversation.realtime_runtime import RealtimeRuntime
from oncue_voice.providers.realtime_models import (
    RealtimeEvent,
    RealtimeSessionOptions,
)
from oncue_voice.providers.realtime_provider import RealtimeProvider


PCM_SAMPLE_RATE_HZ = 24_000
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNEL_COUNT = 1


@dataclass(frozen=True)
class RealtimeEvaluationRequest:
    run_id: str
    variant_id: str
    combination_key: str
    input_text: str
    policy: DialoguePolicy
    session_options: RealtimeSessionOptions
    input_audio: bytes
    artifact_root: Path
    provider: str = "openai"


@dataclass(frozen=True)
class RealtimeEvaluationResult:
    artifact_directory: Path
    succeeded: bool


class RealtimeEvaluationService:
    def __init__(self, provider: RealtimeProvider) -> None:
        self._provider = provider

    async def run(
        self,
        request: RealtimeEvaluationRequest,
    ) -> RealtimeEvaluationResult:
        runtime = RealtimeRuntime(self._provider)
        started_at = time.perf_counter()
        events: list[RealtimeEvent] = []
        first_audio_delta_ms: int | None = None
        async for event in runtime.run(
            request.run_id,
            request.policy,
            self._input_audio(request.input_audio),
            request.session_options,
        ):
            events.append(event)
            if event.type == "audio_delta" and first_audio_delta_ms is None:
                first_audio_delta_ms = round(
                    (time.perf_counter() - started_at) * 1000
                )

        response_audio = b"".join(
            event.audio or b"" for event in events if event.type == "audio_delta"
        )
        succeeded = not any(event.type == "error" for event in events)
        artifact_directory = (
            request.artifact_root / request.run_id / request.variant_id
        )
        artifact_directory.mkdir(parents=True, exist_ok=True)
        self._write_artifacts(
            request,
            artifact_directory,
            events,
            response_audio,
            round((time.perf_counter() - started_at) * 1000),
            first_audio_delta_ms,
            succeeded,
        )
        return RealtimeEvaluationResult(
            artifact_directory=artifact_directory,
            succeeded=succeeded,
        )

    @staticmethod
    async def _input_audio(audio: bytes) -> AsyncIterator[bytes]:
        yield audio

    @classmethod
    def _write_artifacts(
        cls,
        request: RealtimeEvaluationRequest,
        artifact_directory: Path,
        events: list[RealtimeEvent],
        response_audio: bytes,
        elapsed_ms: int,
        first_audio_delta_ms: int | None,
        succeeded: bool,
    ) -> None:
        cls._write_json(
            artifact_directory / "run.json",
            {
                "runId": request.run_id,
                "variantId": request.variant_id,
                "combinationKey": request.combination_key,
                "provider": request.provider,
                "evaluationPath": "realtime",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "succeeded": succeeded,
                "elapsedMs": elapsed_ms,
                "firstAudioDeltaMs": first_audio_delta_ms,
                "audioBytes": len(response_audio),
            },
        )
        cls._write_json(
            artifact_directory / "policy-snapshot.json",
            request.policy.model_dump(mode="json", by_alias=True),
        )
        cls._write_json(
            artifact_directory / "provider-config.json",
            request.session_options.model_dump(
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
                    event.model_dump(
                        mode="json",
                        exclude={"audio"},
                        exclude_none=False,
                    )
                    for event in events
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
    def _evaluation_template(request: RealtimeEvaluationRequest) -> str:
        return f"""# Realtime voice evaluation

- Run: `{request.run_id}`
- Variant: `{request.variant_id}`
- Combination: `{request.combination_key}`
- Path: `realtime`

## Manual scores (1–5)

- Voice quality:
- Persona consistency:
- Scenario goal:
- Interruption handling:
- Safety: pass / violation

## Comments

"""
