# Evidence Limits in Robot-Policy Boundary Conformance

The editable manuscript is `main.tex`, with body, appendix and bibliography alongside it. The standalone PDF is `output/main.pdf`. It uses the unmodified official TMLR assets recorded in `venue-format.json`, in preprint mode. It is a technical report, not submitted or peer reviewed. The existing anonymous draft attribution is preserved; repository account names do not establish manuscript authorship. The public research release does not submit the manuscript to a venue.

The main study reports finite parity and evidence limits. It does not establish a detector advantage, smaller compiled witnesses, learned-policy performance, calibrated physical mappings or hardware safety. The offset-authority ablation distinguishes a weaker pointwise oracle from the two relational methods; this is separate from main-condition parity. The intervention magnitudes, coordinate selection, numerical domain and seeded query construction are stated in the appendix as well as the saved protocol. Human authors retain responsibility for any publication.

The manuscript includes the expanded AI assistance disclosure recorded in `../release/ai-use.json`. Technical checks and AI-assisted reviews are not independent human peer review. `../release/claim-evidence.json` maps reported claims and figures to their inputs, code, saved evidence and verification category.

## Regenerate results and figures

```powershell
uv venv .venv-paper --python 3.13.7
uv pip sync --python .venv-paper/Scripts/python.exe --require-hashes research/publication/reporting/requirements.lock
.venv-paper/Scripts/python.exe research/publication/reporting/build_results.py
```

The generator first verifies original and clean saved evidence without native execution. `generated/facts.json` derives the quantitative claims; `claims.tex`, `outcomes.tex` and two vector/PNG figures are generated from those facts. `generated/manifest.json` binds the input/output and generator bytes. Repeated deterministic regeneration must preserve all generated outputs. Experimental methods and tolerances remain frozen separately; reporting changes do not authorize new native outcomes.

## Compile and inspect

The built-in Codex LaTeX editor compiles this project and provides a live preview. A standalone export also uses the official Tectonic 0.17.0 Windows MSVC executable, downloaded into ignored workspace tooling and verified against the publisher's release-asset SHA-256. `compiler-tool.json` records those identities. Obtain that exact official executable for the portable build script:

```powershell
.venv/Scripts/python.exe research/publication/reporting/build_paper.py --tectonic C:/absolute/path/to/tectonic.exe --allow-downloads
```

Omit `--allow-downloads` after the initial TeX resources are cached; the default is cache-only. The build verifies compiler/style bytes, compiles an immutable source snapshot, checks source stability, retains every build log and records the final PDF hash. `output/latest-build.json` identifies the current successful build. An earlier export failed while source was being edited; its raw logs remain preserved. This has no relationship to the native experimental attempt policy.

Render the final PDF with Poppler, inspect all pages and record the PDF/source/render hashes. The current QA record accompanies the final build. Source compilation, numerical verification and visual inspection are distinct checks.

TMLR is a candidate format, not an asserted acceptance fit. Its [author guide](https://jmlr.org/tmlr/author-guide.html), [criteria](https://jmlr.org/tmlr/acceptance-criteria.html) and [editorial policies](https://jmlr.org/tmlr/editorial-policies.html) were rechecked on 28 October 2026. Before any separately authorized submission, human authors must verify scope/interest, citations, authorship, licensing, anonymization and the venue's current requirements. The private repository has not been released as an anonymous public artifact. The technical and release review is recorded in `../../../validation/publication-review-2026-10-28/`; earlier build and QA records remain historical evidence.
