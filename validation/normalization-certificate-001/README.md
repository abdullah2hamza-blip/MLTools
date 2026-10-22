# Designed normalization-certificate controls

This development evidence contains three independently implemented paper-equation
arithmetic controls and seven genuine native processor controls (four OpenPI,
three LeRobot). Every action is a designed fixture. No demonstrations, model
weights, policy inference, rollout or hardware observations are included.

`inputs.json` contains the complete independently fixed requests.
`lerobot.*` and `openpi.*` preserve actual native requests, stdout/stderr,
parsed boundary arrays, upstream identities and measured runtime/peak RSS.
`*.report.json` contains the separately recomputed finite certificates.
The closed manifest and producer source hashes are in `summary.json`.

The paper-equation profile is an independent reimplementation. The mean/std
LeRobot and epsilon/domain-adjusted OpenPI profiles are labeled extensions,
not reproductions of the paper's decoder or empirical results. There is no
default conformance/safety threshold. See `docs/normalization-certificate.md`.

Data-only replay, also supported after copying this directory:

```sh
python -O scripts/validate_normalization_certificate.py --verify validation/normalization-certificate-001
```
