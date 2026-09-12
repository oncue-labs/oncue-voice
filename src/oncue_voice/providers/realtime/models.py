from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RealtimeEventType = Literal[
    "audio_delta",
    "transcript_delta",
    "transcript_completed",
    "speech_started",
    "speech_stopped",
    "response_completed",
    "error",
]


class RealtimeSessionOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Realtime 세션에서 사용할 speech-to-speech 모델 이름
    model: str
    # 모델이 생성할 음성의 provider voice 식별자
    voice_id: str = Field(alias="voiceId")
    # 모바일에서 보이스 서버로 보낼 PCM 오디오 형식
    input_audio_format: str = Field(default="pcm16", alias="inputAudioFormat")
    # 보이스 서버가 모바일로 내보낼 PCM 오디오 형식
    output_audio_format: str = Field(default="pcm16", alias="outputAudioFormat")
    # PCM 오디오의 초당 샘플 수
    sample_rate_hz: int = Field(default=24_000, alias="sampleRateHz")
    # 사용자가 말을 멈춘 시점을 provider가 자동 감지할지 여부
    turn_detection: Literal["server_vad"] = Field(
        default="server_vad",
        alias="turnDetection",
    )


class RealtimeEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # provider event를 내부에서 구분하는 종류
    type: RealtimeEventType
    # 사용자에게 재생할 음성 조각. 음성이 아닌 event에서는 없음
    audio: bytes | None = None
    # 화면 표시 또는 평가에 사용할 transcript 조각
    text: str | None = None
    # provider 오류나 상태 설명. 비밀값과 원문 오디오는 포함하지 않음
    message: str | None = None
