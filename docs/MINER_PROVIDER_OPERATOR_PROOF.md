# Miner Provider / Operator Proof

This document defines the evidence required before an external miner can be treated as LAUNCH-A execution-grade proof.

## Boundary

This proof does not connect a public production miner, mutate provider state, use endpoint credentials, publish artifacts, restart services, or change production policy.

## Required Evidence

Each external miner proof packet must include:

- provider name and non-secret instance identifier
- operator-approved current or retired state
- code source commit or immutable release tag
- runtime launch mode
- GPU / CPU capability evidence
- reward address in use
- upload/retry/recovery evidence
- timestamp and operator note

## Redaction Rules

Proof output must redact:

- tokens and API keys
- private keys, mnemonics, seed phrases, and wallet secret material
- SSH private keys
- provider billing secrets
- endpoint credentials
- bearer/basic auth values

## Current Gate

Local source can prove recovery code exists, but it cannot prove the provider/operator state of a remote miner. Until provider evidence or an operator-signed retirement/replacement note exists, external miner proof remains `NO-GO`.
