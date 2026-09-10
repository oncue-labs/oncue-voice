# 평가 notebook 사용법

기본 실행은 fake provider를 사용하므로 API 비용이 없다.

```bash
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

실제 OpenAI 평가를 실행할 때는 합성 음성 파일과 환경 변수를 지정한다.

```bash
export ONCUE_EVALUATION_PROVIDER=openai
export OPENAI_API_KEY=...
export ONCUE_EVALUATION_AUDIO_PATH=/path/to/synthetic-input.wav
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

실행 결과는 `notebooks/artifacts/<runId>/<variantId>/`에 저장한다. API key와 같은 비밀값이나 실제 개인정보는 입력·정책·artifact에 넣지 않는다.
