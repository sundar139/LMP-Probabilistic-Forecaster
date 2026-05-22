# Engineering notes

## 2026-05-22 — Live PJM one-year backfill rate/partition tuning

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
