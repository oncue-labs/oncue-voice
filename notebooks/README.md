# 평가 notebook 사용법

기본 실행은 fake provider를 사용하므로 API 비용이 없다.

```bash
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

Realtime speech-to-speech 평가 notebook은 다음처럼 실행한다.

```bash
poetry run jupyter lab notebooks/realtime_persona_scenario_evaluation.ipynb
```

실제 OpenAI 평가를 실행할 때는 `.env.example`을 복사해 `.env`를 만들고 값을 입력한다. `.env`는 Git에 포함되지 않으며 notebook이 자동으로 읽는다. 저장소의 합성 입력 fixture는 `tests/fixtures/evaluation/input.wav`와 `tests/fixtures/evaluation/input.pcm`이다. 실행하는 notebook에 맞춰 `ONCUE_EVALUATION_AUDIO_PATH`를 선택한다. 실행 후 실제로 사용된 입력은 해당 평가 회차의 `input/` 아래에 복사된다.

```bash
cp .env.example .env
# .env에서 OPENAI_API_KEY와 ONCUE_EVALUATION_AUDIO_PATH를 입력한다.
```

분리형 pipeline 평가:

```bash
poetry run jupyter lab notebooks/voice_persona_scenario_evaluation.ipynb
```

Realtime 평가도 같은 `.env`를 사용한다. `OPENAI_REALTIME_MODEL`과 선택한 voice 환경 변수는 `.env`에서 변경할 수 있다.

```bash
poetry run jupyter lab notebooks/realtime_persona_scenario_evaluation.ipynb
```

실행 결과는 `notebooks/evaluations/<combinationKey>/<runId>/`에 저장한다. 각 회차 아래의 `input/`에는 실제 사용한 입력과 `input.json`을, `artifacts/`에는 정책·provider 설정·transcript·응답 음성·수동 평가를 저장한다. 설정값을 바꿔 다시 실행하면 같은 통화 조합 아래에 다음 `run-002`, `run-003` 폴더가 생성된다. API key와 같은 비밀값이나 실제 개인정보는 입력·정책·artifact에 넣지 않는다.
