# 평가 notebook 사용법

기본 실행은 fake provider를 사용하므로 API 비용이 없다.

```bash
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

Realtime speech-to-speech 평가 notebook은 다음처럼 실행한다.

```bash
poetry run jupyter lab notebooks/realtime_persona_scenario_evaluation.ipynb
```

실제 OpenAI 평가를 실행할 때는 합성 음성 파일과 환경 변수를 지정한다.

```bash
export ONCUE_EVALUATION_PROVIDER=openai
export OPENAI_API_KEY=...
export ONCUE_EVALUATION_AUDIO_PATH=/path/to/synthetic-input.wav
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

Realtime 평가를 실제 OpenAI로 실행할 때는 `OPENAI_REALTIME_MODEL`과 선택한 voice 환경 변수도 지정할 수 있다.

```bash
export ONCUE_EVALUATION_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_REALTIME_MODEL=gpt-realtime
export ONCUE_EVALUATION_AUDIO_PATH=/path/to/synthetic-input.pcm
poetry run jupyter lab notebooks/realtime_persona_scenario_evaluation.ipynb
```

실행 결과는 `notebooks/artifacts/<runId>/<variantId>/`에 저장한다. API key와 같은 비밀값이나 실제 개인정보는 입력·정책·artifact에 넣지 않는다.
