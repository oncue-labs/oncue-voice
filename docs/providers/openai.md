# OpenAI provider

OnCue MVP의 기준 Public provider다. `ConversationRuntime`은 OpenAI SDK를 직접 호출하지 않고, 다음 세 adapter를 통해 STT·LLM·TTS를 각각 사용한다.

## Adapter 구성

| 기능 | OpenAI API | 초기 모델 | adapter 출력 |
| --- | --- | --- | --- |
| LLM | Chat Completions streaming | `gpt-5-mini` | `AssistantChunk` |
| STT | Audio Transcriptions streaming | `gpt-4o-mini-transcribe` | `TranscriptSegment` |
| TTS | Speech streaming response | `gpt-4o-mini-tts` | PCM bytes |

- LLM adapter는 대화 정책을 system message로 전달하고 사용자 발화를 user message로 전달한다.
- STT adapter는 현재 한 번의 발화 오디오를 메모리에서 하나의 파일 입력으로 만든 뒤 streaming transcription 응답을 받는다. OpenAI의 파일 기반 transcription API를 사용하는 1차 구현이며, WebRTC 음성을 작은 단위로 직접 전송하는 실시간 STT 최적화는 별도 작업이다.
- TTS adapter는 `pcm` 출력과 voice ID를 사용한다. MVP에서 지원하는 조정 값은 `speed`와 `instructions`이며, 지원하지 않는 값은 factory가 거부한다.

## 설정

`OPENAI_API_KEY` 환경 변수에서만 자격 증명을 읽는다. API key, token, password는 로그와 평가 artifact에 기록하지 않는다.

`ProviderSettings` 예시는 다음과 같다.

```json
{
  "provider": "openai",
  "llmModel": "gpt-5-mini",
  "sttModel": "gpt-4o-mini-transcribe",
  "ttsVoiceId": "alloy",
  "ttsOptions": {
    "speed": 1.0,
    "instructions": "Speak warmly."
  }
}
```

현재 factory가 검증하는 분리형 OpenAI 모델·voice와 옵션 목록은 코드의 `ProviderCapabilities`에 정의한다. 실제 모델 응답 품질과 비용은 notebook에서 같은 입력을 사용해 평가한다.

## Realtime speech-to-speech adapter

Realtime 평가는 `OpenAiRealtimeProvider`가 OpenAI Realtime WebSocket 세션을 `RealtimeProvider` 계약으로 감싼다. 모바일은 OpenAI에 직접 연결하지 않고, 이후 WebRTC bridge가 `oncue-voice`의 Realtime session과 연결한다.

- `session.update`에 최종 `DialoguePolicy`와 voice·PCM·turn detection 설정을 전달한다.
- 모바일에서 받은 PCM 오디오 조각은 `input_audio_buffer.append` event로 보낸다.
- `response.output_audio.delta`를 내부 `audio_delta`로, output transcript를 `transcript_delta`로 변환한다.
- input transcript는 평가용 보조 정보이며, 응답을 만들기 위한 별도 STT 단계로 사용하지 않는다.
- provider 고유 event object와 API 오류는 adapter 밖으로 내보내지 않는다.

Realtime adapter는 현재 연결 가능성과 평가용 event 변환을 확인하는 단계다. 실제 운영 provider와 WebRTC 연결은 수동 품질 평가와 통합 테스트가 통과한 뒤 결정한다.

## 참고 API

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI Realtime API reference](https://platform.openai.com/docs/api-reference/realtime)
- [OpenAI Python Realtime example](https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py)
- [Chat Completions streaming](https://github.com/openai/openai-python/blob/main/examples/streaming.py)
- [Speech-to-text API](https://github.com/openai/openai-python/blob/main/examples/speech_to_text.py)
- [Text-to-speech streaming](https://github.com/openai/openai-python/blob/main/examples/text_to_speech.py)
