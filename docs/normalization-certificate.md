# Normalization-mismatch certificate baseline

`scripts/normalization_certificate.py` is an independent implementation of the
finite-action certificate in [Tai, arXiv:2606.03724v1, §§4.1–4.3](https://arxiv.org/html/2606.03724v1).
It is not the author's code or a reproduction of the paper's experimental data.
For the paper-equation profile, a masked coordinate decodes as
`q01 + z * (q99 - q01)` and an unmasked coordinate returns `z`. Both decoders
receive the same normalized sequence. The report includes per-sample L2 drift,
its mean, percentiles, inclusive `distance >= threshold` fractions, mask
disagreement and explicitly identified gripper coordinates. Physical input
actions are normalized with the intended metadata first. Reporting thresholds
are not safety margins.

## Explicit profiles

The input schema requires an explicit profile; their coordinate domains and
numerical behavior are different:

| Profile | Forward normalization | Decoding |
| --- | --- | --- |
| `paper_quantile_equations` | `(a-q01)/(q99-q01)` on masked coordinates | `q01 + z*(q99-q01)` on masked coordinates |
| `openpi_quantile_extension` | `2*(a-q01)/(q99-q01+1e-6)-1` | `(z+1)/2*(q99-q01+1e-6)+q01` |
| `lerobot_mean_std_extension` | `(a-mean)/(std+1e-8)` | `z*std+mean` |

The first row uses the paper equations. The latter two are separately labeled
extensions for the pinned native implementations. OpenPI requires quantile
statistics for every coordinate in this profile; trailing padded-coordinate
behavior is outside its scope. LeRobot's profile applies MEAN_STD to the entire
action vector. No arbitrary per-coordinate normalization mask is invented for
these two native profiles. Normalization masks are distinct from relative-action
masks. Zero standard deviations are supported for the LeRobot extension, whose
forward and inverse operations can then be lossy.

The native formulas are independently checked against the pinned
[OpenPI transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py)
and [LeRobot normalization processor](https://github.com/huggingface/lerobot/blob/5517a326eeb9b457cfe07d59b9bfdf291799c8d9/src/lerobot/processor/normalize_processor.py).
The two environments execute sequentially in isolated workers. Selected upstream
source identities are verified before and after execution. Recorded float64
inputs, normalized values, and both native decoded arrays are compared with
the independent implementation, with absolute/relative allowances of `1e-12`.
Those allowances apply to these measured controls, not arbitrary upstream
arithmetic. No model, tokenizer, weights, simulator or robot is constructed.

## Input, output and abstention

The schema version is 1. Inputs explicitly name coordinates and units, intended
and candidate metadata authorities, calibration kind/space/authority, selected
metric indices and their common unit, gripper indices, thresholds and percentiles.
Gripper identity is never inferred from the last array position. See
`validation/normalization-certificate-001/inputs.json` for complete examples.

The API is `certificate(request)`. Its standalone data-only CLI is:

```sh
python scripts/normalization_certificate.py request.json --output report.json
```

The output destination must be new. The loader rejects duplicate JSON keys and
nonfinite constants. Sample/dimension and file-size bounds are explicit.

The report retains the request's canonical hash, normalized inputs, both decoded
sequences, per-sample/per-axis differences and per-axis gripper diagnostics.
These arrays are raw numerical diagnostics. The selected aggregate metric is
`UNKNOWN` when authority is missing or selected coordinate units do not all
match the explicit metric unit; aggregate distances and thresholds are then
withheld. A synthetically labeled coordinate unit is not physical calibration.
Authority strings remain caller assertions, not authenticated provenance.

An available metric has status `COMPUTED`, not PASS or FAIL. There is no default
detection threshold. A final evaluation must freeze any descriptive threshold or
decision rule separately and apply it fairly to relevant methods. Even an
identity certificate has tail fraction 1 at threshold zero because the comparison
is inclusive. If normalization is lossy, drift is measured against the intended
decoded sequence, not against the original physical inputs.

Implementation choices are explicit: no clipping; linear percentile interpolation
at `(n-1)*p`; binary64 arithmetic; direct subtraction of the two decoders when
masks differ. The output is a finite normalization-boundary calculation. It
does not certify deployment support, controller composition, chunk state,
task performance or safety.

## Development evidence and reproduction

`validation/normalization-certificate-001/` contains ten fixed designed controls:
three paper-equation arithmetic controls, four native OpenPI controls and three
native LeRobot controls.
They test identity, a decoder intervention, mask change, out-of-range normalized
inputs, and a lossy LeRobot forward path. These actions are designed fixtures,
not demonstration-derived calibration or research samples from policy rollouts.
Hand-calculated tests independently check the arithmetic. No published result
table or replay-success experiment is reproduced here.

```sh
python scripts/validate_normalization_certificate.py --lerobot-python /path/to/lerobot/python --openpi-python /path/to/openpi/python --output validation/normalization-certificate-new
python -O scripts/validate_normalization_certificate.py --verify validation/normalization-certificate-001
```

The output directory must be new. Verification also works on a copied evidence
directory without either ML library installed. It regenerates all fixed requests,
checks the closed file manifest, binds raw stdout and parsed captures to the
input/source snapshots, verifies native parity, and recalculates every report.
Worker timeouts are 180 seconds. Peak RSS is measured and checked after execution
against 2 GiB; this is not an operating-system memory ceiling. A Python audit hook
rejects network connections and is not an operating-system sandbox.

The existing ACT/SmolVLA MEAN_STD metadata can only use the separately labeled
extension; the paper quantile profile cannot silently substitute for it.
Using designed probe actions on any profile does not establish the calibration
distribution assumed by a demonstration-based evaluation. The baseline remains
development work until the main corpus and evaluation are frozen and executed.
