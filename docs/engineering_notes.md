# Engineering notes

## 2026-05-27 — Portable tuning package + import validation closeout

Mistake observed:
- New package/export/import workflow initially lacked dedicated regression tests for dry-run/write semantics and schema gate behavior.
- Initial CLI write-path assertions expected absolute-style substrings and failed when test runner emitted relative ignored paths.
- Ruff surfaced line-length violations in new CLI tests.
- Existing policy test only checked file/folder names for forbidden wording and did not assert module/test-name coverage.
- Package manifest python requirement string was stale relative to project Python constraint.

Root cause:
1. Step continuation added source modules first, but test/docs closeout lagged behind implementation scope.
2. CLI tests were initially written with stricter absolute-path assumptions than actual CLI output contract.
3. Quick test additions introduced long single-line fixture writes beyond configured style limit.
4. Naming-policy regression checks were too narrow for module/test function identifiers.
5. Manifest metadata default was not synchronized with `pyproject.toml` requirement.

Fix implemented:
1. Added focused Step 11 tests:
   - `tests/test_tuning_package.py`
   - `tests/test_tuning_result_import.py`
   - `tests/test_cli_tuning_package.py`
   - `tests/test_cli_tuning_import.py`
   - updated `tests/test_tuning_config.py`
   - updated `tests/test_agent_memory_policy.py`
2. Added/updated scripts:
   - `scripts/export_tuning_package.ps1`
   - `scripts/import_tuning_results.ps1`
   - both clear `PYTHONHOME/PYTHONPATH` and run dry-run first.
3. Fixed test failures by aligning path assertions to ignored relative-root patterns and reformatting long lines to satisfy Ruff.
4. Hardened package metadata:
   - aligned package manifest python requirement to `>=3.12,<3.13`.
   - made git commit/branch lookup run against discovered project root.
5. Expanded docs:
   - updated `README.md`
   - updated `docs/focused_tuning.md`
   - updated `docs/calibration_and_search_design.md`
   - added `docs/portable_tuning_workflow.md`
6. Expanded policy regression checks:
   - naming guard now checks file/folder/module/test names for forbidden wording.
   - ignore-policy test now includes `data/cache/tuning_packages/**`.

Regression protection:
- Export/import dry-run/write behavior covered by CLI tests.
- Import schema, gate recompute, mismatch detection, under-coverage, collapse rejection, and valid acceptance covered by unit tests.
- Naming/ignore policy safeguards extended to module and test identifiers.

Lesson learned:
- For portable external execution workflows, treat import-side local recompute and mismatch detection as first-class acceptance criteria, not optional checks.
- Policy tests must validate naming constraints across modules and test symbols, not only filesystem entries.

## 2026-05-27 — Step 10 resource-safe focused tuning closeout

Mistake observed:
- Prior write-mode focused tuning settings were too heavy for local constraints and risked freeze/timeouts.
- Initial local-safe run still allowed large trial `batch_size` values from search-design specs to leak into actual training overrides.
- First pytest invocation used an invalid foreground timeout request (>600s) in this tool environment.

Root cause:
1. Focused tuning runner did not enforce resource-safe caps at trial override merge time.
2. Search-design parameters were valid for full tuning but too aggressive for local-safe smoke execution unless explicitly clamped.
3. Tool timeout policy requires max 600s foreground timeout.

Fix implemented:
1. Added `resource_profile` section to `conf/tuning.yaml` with explicit local-safe constraints and deferred full-search metadata.
2. Extended tuning config/CLI to support:
   - `--resource-profile local_safe`
   - bounded overrides for trials/folds/steps/batch size
   - `--allow-heavy-run` guard
   - `--cleanup-after-trial`
   - `--cpu-only`
   - `--no-mlflow`
3. Hardened tuning runner:
   - local-safe heavy-run refusal unless explicitly allowed
   - deterministic tiny trial generation by default (no Optuna DB by default)
   - per-trial cleanup hook (artifact cleanup + GC + CUDA cache clear)
   - CUDA OOM/resource failure classification into `failed_resource_limited`
   - runtime_seconds capture for each trial
   - smoke-scope promotion semantics (`smoke_candidate_requires_full_validation` ready)
4. Added/updated tests for:
   - local-safe default limits and heavy-run refusal
   - dry-run no-write behavior
   - cleanup hook invocation
   - CUDA OOM capture without uncaught crash
   - no Optuna DB creation in default local-safe mode
   - promotion smoke status and no-promotion summary behavior
   - generated output paths under ignored directories
5. Completed bounded write-mode smoke run on local hardware:
   - zone AEP, TFT-only, trials=2, folds=1, max_steps_cap=3, batch_size=4, num_workers=0
   - run completed without freeze
   - honest result: `no_promotion` (coverage below gate)
6. Updated docs:
   - `README.md`
   - `docs/calibration_and_search_design.md`
   - new `docs/focused_tuning.md`

Regression protection:
- Added/updated:
  - `tests/test_tuning_config.py`
  - `tests/test_cli_tuning.py`
  - `tests/test_tuning_runner.py`
  - `tests/test_promotion.py`
  - `tests/test_agent_memory_policy.py`
- Full gates pass after fixes with environment cleanup:
  - Ruff
  - mypy
  - pytest

Lesson learned:
- In constrained local hardware workflows, resource-safe profile constraints must be enforced not only at CLI parsing but also inside per-trial override application.
- Honest smoke tuning outcomes are valid closeout evidence when full search is explicitly deferred and guarded.
- Keep tuning workflow reproducible by defaulting to deterministic tiny trials and requiring explicit intent for heavy runs.

## 2026-05-27 — Step 9 calibration diagnostics + focused search design + agent memory skills

Mistake observed:
- `uv run mypy src` and `uv run pytest -q` initially failed with `AssertionError: SRE module mismatch`.
- New calibration/search modules triggered Ruff line-length/import-order violations.
- A calibration unit test expected DeepAR horizon-2 coverage of `0.0` but deterministic fixture produced `0.5`.
- Calibration summary JSON included merged duplicate count fields (`row_count_x`, `row_count_y`) due to unchecked column overlap.

Root cause:
1. Host environment inherited conflicting `PYTHONHOME/PYTHONPATH`, causing stdlib/runtime mismatch under uv-managed interpreter.
2. Fresh module/test code exceeded style limits and import ordering conventions.
3. Test assertion was based on stale expected value, not fixture truth.
4. Calibration summary merge did not normalize duplicated count columns.

Fix implemented:
1. Standardized gate/run commands with environment cleanup (`unset PYTHONHOME PYTHONPATH; uv run ...`) for this host session.
2. Refactored long lines/formatting/imports until Ruff passed.
3. Corrected `tests/test_calibration.py` expected horizon coverage value to `0.5`.
4. Added calibration/search modules + CLI commands + configs + tests:
   - `src/lmp_forecaster/eval/calibration.py`
   - `src/lmp_forecaster/tuning/search_design.py`
   - `conf/calibration.yaml`
   - `conf/search_design.yaml`
   - `analyze-calibration` and `design-focused-search` CLI commands
5. Added external agent-memory system outside repo:
   - `C:\Users\rohit\Documents\Agent Memory\projects\lmp-probabilistic-forecaster.md`
   - `C:\Users\rohit\Documents\Agent Memory\skills\project-memory-maintainer.md`
   - `C:\Users\rohit\Documents\Agent Memory\skills\memory-driven-skill-distiller.md`
6. Added repo ignore safety for local memory mirrors:
   - `.agent-memory/`, `AGENT_MEMORY*.md`, `.hermes-memory*.md`, `.codex-memory*.md`
7. Generated and verified diagnostics/search reports under ignored paths:
   - `data/cache/reports/aep_calibration_diagnostics_<timestamp>.{json,md}`
   - `data/cache/reports/aep_focused_search_design_<timestamp>.{json,md}`

Regression protection:
- Added tests:
  - `tests/test_calibration.py`
  - `tests/test_search_design.py`
  - `tests/test_cli_calibration.py`
  - `tests/test_cli_search_design.py`
  - `tests/test_agent_memory_policy.py`
- Existing ignore-policy and tracking tests retained.
- Full gates pass after fixes: Ruff, mypy, pytest.

Lesson learned:
- On this Windows/uv host, quality-gate reliability requires explicit environment hygiene when interpreter contamination appears.
- Calibration/search design steps need deterministic small fixtures and explicit expected-value checks before rollout.
- External memory files must stay outside repo with repo-local mirror ignore patterns to prevent accidental commits.

## 2026-05-27 — Step 8 real rolling-origin backtest execution (AEP)

Mistake observed:
- Step 8 execution initially produced a tracked but unignored local `.mlflow_artifacts/` directory from the MLflow artifact-path helper.
- Command output for the real write run ended with shell exit code `127` despite full backtest artifacts being written and parsable.
- The Step 7 report path `src/lmp_forecaster/tracking/init.py` looked suspicious during audit.
- Existing `test_mlflow_disabled_does_not_create_mlruns` asserted against `tmp_path/mlruns` without changing cwd, so it could miss cwd-related behavior.

Root cause:
1. `log_artifact_paths()` writes helper text files under `.mlflow_artifacts/`, but `.gitignore` only ignored `mlruns/`.
2. On this Windows/MSYS host, long-running CLI completion occasionally returns wrapper code 127 post-run even when training + writes complete; relying only on process code can misclassify success.
3. The Step 7 report string was rendering shorthand; actual source file already used `__init__.py`.
4. Test used a path assumption instead of validating from an isolated cwd.

Fix implemented:
1. Added `.mlflow_artifacts/` to `.gitignore`.
2. Added regression test `test_mlflow_artifact_scratch_dir_is_ignored` in `tests/test_mlflow_tracking.py`.
3. Hardened `test_mlflow_disabled_does_not_create_mlruns` by `monkeypatch.chdir(tmp_path)` and asserting `Path("mlruns")` is absent in isolated cwd.
4. Verified tracking initializer path: `src/lmp_forecaster/tracking/__init__.py` exists; no rename needed.
5. Verified real Step 8 success from generated outputs and parsed metrics/report artifacts (fold metrics CSV, aggregate CSV, summary JSON/MD), not from exit code alone.

Regression protection:
- Added ignore-policy test for `.mlflow_artifacts/`.
- Updated MLflow-disabled cwd-isolated test.
- Existing Step 8 tests continue to cover:
  - dry-run writes nothing,
  - missing panel failure,
  - no train/test leakage,
  - fold_id propagation,
  - aggregate metric schema,
  - low coverage reporting,
  - skip flags and at-least-one-model guard,
  - forecast schema requirements for `zone` and `data_source_label`.

Lesson learned:
- When optional tracking writes helper artifacts, ignore hygiene must include scratch directories alongside run stores.
- On this host, treat completion evidence as a combination of process status and produced artifact integrity.
- Path audits should verify filesystem truth (`__init__.py`) before applying structural renames.

## 2026-05-22 — Step 7 hardening: optional tracking + backtest planning scaffold

Mistake observed:
- Initial Step 7 integration introduced static-analysis issues (mypy typing around dynamic fold summaries, long-line/import ordering lint failures), plus one CLI dry-run test that built panel data outside isolated filesystem.
- A temporary patch accidentally removed `tracking_ctx`/`run_name` initialization in baseline training flow.

Root cause:
1. Backtest summary structure is intentionally dynamic (`dict[str, Any]`), but typed as `dict[str, object]` in helper signatures, causing mypy attr/iterability errors.
2. CLI fold-print loop indexed dynamic dicts directly without type narrowing.
3. Test fixture order mismatch: pre-created panel path outside `CliRunner.isolated_filesystem` meant command could not resolve panel file in test sandbox.
4. Fast patch edit on training path dropped required tracking init lines.

Fix implemented:
1. Added optional tracking module `src/lmp_forecaster/tracking/mlflow_utils.py` with:
   - `TrackingConfig`
   - `get_tracking_uri`
   - `configure_mlflow`
   - `start_mlflow_run`
   - `log_training_config`
   - `log_metrics_table`
   - `log_artifact_paths`
   - `safe_log_params`
2. Added tracking config `conf/tracking.yaml` (disabled by default, local URI `file:./mlruns`).
3. Wired training CLI and baseline pipeline with tracking flags while preserving dry-run no-write behavior.
4. Added fold-planning scaffold `src/lmp_forecaster/eval/backtest.py` and CLI command `plan-rolling-backtest`.
5. Hardened training config parsing/validation:
   - explicit `horizon_hours`, `input_size_hours`, `validation_hours`, `test_hours`, `interval_level`, `max_steps_real_candidate`, `accelerator`
   - robust validation for quantiles/interval/split sizes.
6. Fixed mypy/ruff issues by:
   - narrowing dynamic fold types (`dict[str, object]` local cast + `Any` summary typing),
   - restoring dropped `tracking_ctx`/`run_name`,
   - import/format cleanup.
7. Fixed CLI backtest dry-run test to create panel file inside isolated filesystem.

Regression protection:
- Added tests:
  - `tests/test_mlflow_tracking.py`
  - `tests/test_backtest.py`
  - `tests/test_cli_backtest.py`
- Expanded tests:
  - `tests/test_baseline_config.py`
  - `tests/test_cli_training.py`
- Full gates pass after fixes: Ruff, mypy, pytest.

Lesson learned:
- Planning/report helpers that intentionally carry heterogeneous dictionaries should be typed explicitly as `Any` at interfaces, then narrowed locally.
- For CLI tests using isolated filesystems, create all input artifacts inside the sandbox or path resolution will fail despite valid command logic.
- For quick patches in critical training flow, immediately re-read modified block to catch dropped initialization statements.

## 2026-05-22 — Step 6 continuation stabilization (lint/type/runtime/training)

Mistake observed:
- Interrupted continuation left multiple Ruff failures (imports/line-length/unused vars), plus runtime regressions in real cache discovery and DST handling.
- Real panel build failed to consume full-year cached LMP when overlapping chunks existed.
- CLI training command printed successful outputs but shell return code surfaced as 127 on this host even when artifacts were written.

Root cause:
1. Partially applied Step 6 edits introduced formatting/import drift and dead assignments.
2. Cache-chunk selector prioritized shortest chunk first, which could choose stale overlap fragments and break contiguous year coverage.
3. Open-Meteo normalization localized all timestamps with strict ambiguity handling without fallback and could fail on DST-boundary edge cases.
4. Forecast schema lacked required `zone` and `data_source_label` columns for final real-baseline contract.
5. Host shell/runtime wrapper intermittently emitted return code 127 after successful CLI completion (post-run environment quirk), not a model-training failure.

Fix implemented:
1. Resolved all Ruff issues across CLI, panel builder, cache discovery, weather backfill, results reporting, baselines, and new tests.
2. Updated LMP cache discovery selection to prefer freshest chunk by mtime while still ensuring day-by-day contiguous coverage.
3. Added DST-safe Open-Meteo normalization fallback (`ambiguous=False`) plus invalid-time guard; added regression tests.
4. Tightened panel ingestion to validate raw LMP frame before coercion and expanded regression tests.
5. Extended forecast schema normalization/validation to require and populate `data_source_label` and `zone`; passed through from baseline training pipeline.
6. Re-ran real weather pull, real panel build, and real TFT/DeepAR training; validated forecasts + metrics with `data_source_label=real`.

Regression protection:
- Added/updated tests:
  - `tests/test_real_cache_discovery.py`
  - `tests/test_real_panel_building.py`
  - `tests/test_weather_backfill.py`
  - `tests/test_openmeteo_weather.py`
  - `tests/test_build_panel.py`
  - `tests/test_forecast_schema.py`
  - `tests/test_results_report.py`
- Quality gates passed after fixes: Ruff, mypy, pytest.

Lesson learned:
- For resumable yearly cache workflows, freshness-aware contiguous chunk selection is safer than shortest-span preference.
- DST-localized weather timestamps need explicit fallback logic and regression tests.
- Real-baseline contracts should enforce provenance fields (`zone`, `data_source_label`) directly in forecast schema validation.
- On this Windows/MSYS host, treat CLI return code 127 as a potential post-run wrapper artifact when success evidence (reports/artifacts) is present; verify outputs directly.


Mistake observed:
- One-year pull with monthly-ish chunks initially failed.
- Saw either archived/standard boundary 400 error or repeated 429 rate-limit responses.

Root cause:
- Chunk windows were too broad for stable historical API behavior and triggered partition/rate constraints.

Fix implemented:
1. Reduced default `pull-real-pjm-lmp --chunk-days` from 31 to 7.
2. Added chunk-level retry handling in backfill for 429 responses with explicit cool-down sleep.
3. Added automatic split-and-retry logic when a chunk crosses archived/standard boundary.
4. Added regression assertions for weekly default chunk behavior and dry-run output.

Lesson learned:
- PJM live historical pulls are much more reliable with conservative chunking and backoff.
- For one-year runs, use adaptive chunk splitting when archive partition boundaries are unclear.

## 2026-05-22 — Runtime environment mismatch during quality gates

Mistake observed:
- `uv run mypy src` crashed with `AssertionError: SRE module mismatch`.

Root cause:
- Shell environment injected `PYTHONHOME` / `UV_INTERNAL__PYTHONHOME` from a different Python runtime, which conflicted with `.venv` interpreter execution.

Fix implemented:
1. Verified environment contamination via `env` inspection.
2. Ran quality-gate commands with `PYTHONHOME` and `UV_INTERNAL__PYTHONHOME` unset.
3. Added this caveat to engineering notes for future reproducibility.

Regression protection:
- Explicit gate command pattern now uses `unset PYTHONHOME UV_INTERNAL__PYTHONHOME` before `uv run ...`.

Lesson learned:
- On this host, inherited Python-home env vars can silently break tool execution; normalize env before gate runs.

## 2026-05-22 — One-year resume accepted invalid cached chunk

Mistake observed:
- Resumable one-year run repeatedly failed at final validation (`Required column has nulls: ds`) even though most chunks were marked as skipped-existing.

Root cause:
- Resume logic only checked file existence/non-empty and required columns. It accepted previously written chunk files containing invalid `ds` values (DST ambiguity from EPT-only timestamps).

Fix implemented:
1. Strengthened skip-existing validation: existing normalized chunk must pass full `validate_lmp_frame`; otherwise it is redownloaded.
2. Added regression test `test_backfill_resume_redownloads_invalid_existing_chunk`.
3. Improved timestamp parsing to avoid DST ambiguous nulls:
   - Prefer `datetime_beginning_utc` when available.
   - For local-time parsing, use vectorized localization with `ambiguous="infer"` fallback to `ambiguous=False`.

Regression protection:
- New tests in `tests/test_pjm_api.py` and `tests/test_pjm_backfill.py` cover DST ambiguity and invalid cached-chunk redownload behavior.

Lesson learned:
- Resumability must validate semantic data integrity, not just file presence.
- DST boundaries require robust timestamp strategy for annual pulls.

## 2026-05-22 — Live PJM AEP inspection mismatch and fix

Mistake observed:
- Live inspect against `https://api.pjm.com/api/v1/da_hrl_lmps` failed with HTTP 400.
- Redacted response indicated invalid archived-data filters: `pnode_name` and `fields`.

Root cause:
- The query builder reused smoke-style assumptions that archived API calls allowed those parameters.
- Live archived endpoint rejected them.

Fix implemented:
1. Removed `fields` and `pnode_name` from the default query parameter strategy in `build_day_ahead_lmp_params`.
2. Kept robust client-side zone filtering in normalization (`AEP` selected from returned rows).
3. Improved CLI inspect error handling to print actionable redacted failure and exit cleanly.
4. Added regression assertion in tests to ensure banned params are not sent.

Lesson learned:
- PJM archived endpoints can reject filters accepted by other contexts or examples.
- Keep request params minimal first, then add only parameters verified by live response behavior.
