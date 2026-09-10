class ProviderError(RuntimeError):
    """외부 음성 provider 호출에서 공통으로 사용하는 오류."""


class ProviderAuthenticationError(ProviderError):
    """provider 인증이 거부된 경우의 오류."""


class ProviderRateLimitError(ProviderError):
    """provider 요청 한도를 초과한 경우의 오류."""


class ProviderTimeoutError(ProviderError):
    """provider 응답이 제한 시간 안에 오지 않은 경우의 오류."""


class ProviderRequestError(ProviderError):
    """provider 요청이 연결 또는 서버 오류로 실패한 경우의 오류."""


class ProviderResponseError(ProviderError):
    """provider 응답 형식이 공통 계약으로 변환되지 않은 경우의 오류."""
