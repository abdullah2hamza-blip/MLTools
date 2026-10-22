# Fresh ActionABI setup and native-control reproduction

This run executed the documented `scripts/prepare_actionabi.py` setup end to end using a new `.venv-actionabi-clean`, hash-locked CMake/Ninja installation, fresh pinned source checkouts, a newly downloaded SHA-verified official portable compiler archive and a new build directory. No previous build object or installed native library was reused.

The complete stock CPU core and CLI compiled. Both upstream CPU package/CLI test groups passed. All 138 main-repository, 250 CLI11 and 1,090 JSON checkout files match their pinned Git blob bytes exactly; the preparation command sets LF checkout behavior explicitly. Source identities are checked before and after the 17 native driver invocations.

All **55** fixture, request, native stdout and native stderr files match `validation/actionabi-native-001/` byte for byte. Build commands/logs use the fresh paths and have their own identities. Raw source inventory fingerprints differ from the first run's Windows line-ending conversions, as explicitly recorded. Binary hashes are retained rather than assuming reproducible PE executable bytes.

`summary.json` binds the 68 recorded experiment files, project/profile sources, native binary, compiler archive, source inventories and timings. The semantic scope is identical to the first run: synthetic development trajectory/grammar controls, no physical conformance, empirical coverage, task performance or safety claim. Replication does not create additional independent cases or research-held-out evidence.

Optimized data-only verification:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_actionabi.py --verify validation/actionabi-native-reproduction-001
```

This completes a clean reproduction of this optional baseline workflow. The eligible corpus, final protocol/experiments, main research reproduction and complete paper remain unfinished.
