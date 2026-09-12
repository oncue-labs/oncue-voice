# OnCue coturn reference

`coturn.conf` contains only non-secret defaults. The orchestration deployment
must provide the realm, listener/relay ports, and the same TURN credential
values to coturn and `oncue-voice`.

The voice service reads these variables:

- `ONCUE_VOICE_TURN_URLS`: comma-separated TURN URLs, for example
  `turn:coturn:3478?transport=udp,turn:coturn:3478?transport=tcp`.
- `ONCUE_VOICE_TURN_USERNAME` and `ONCUE_VOICE_TURN_CREDENTIAL`: TURN
  credentials supplied through the deployment environment.
- `ONCUE_VOICE_TURN_USERNAME_FILE` and
  `ONCUE_VOICE_TURN_CREDENTIAL_FILE`: Docker Secret file alternatives. A
  direct value takes precedence over its file variable.

The coturn container must be started with the same username and credential
using its deployment secret mechanism, for example the equivalent of
`--user <username>:<credential>`. Do not put those values in this repository's
configuration file or logs.

The deployment-level coturn variables are `ONCUE_TURN_REALM`,
`ONCUE_TURN_LISTENING_PORT`, `ONCUE_TURN_TLS_LISTENING_PORT`,
`ONCUE_TURN_MIN_PORT`, and `ONCUE_TURN_MAX_PORT`. They configure the coturn
container; the voice `ONCUE_VOICE_TURN_URLS` value must point to the resulting
listener addresses.

The reference ports are UDP/TCP `3478` for TURN and TCP `5349` for TLS TURN,
with relay ports `49152-65535`. If a deployment changes them, it must update
both the coturn command/configuration and `ONCUE_VOICE_TURN_URLS`.
