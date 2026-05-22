# Real PJM ingestion

## Why synthetic came first

Earlier steps used synthetic and smoke inputs to validate architecture safely before live market ingestion was stabilized.

## Why real PJM ingestion starts now

This step adds production-style PJM API support for real AEP day-ahead hourly LMP pulls with chunking, retries, rate limiting, and quality reporting.

## API key setup

Add to `.env`:

- `PJM_API_KEY=<your key>`
- `PJM_API_BASE_URL=https://api.pjm.com/api/v1`
- `PJM_DATA_MINER_BASE_URL=https://dataminer2.pjm.com`

The API key is never printed in clear text by CLI diagnostics.

## Endpoint and parameter strategy

Primary endpoint:

- `https://api.pjm.com/api/v1/da_hrl_lmps`

Core query params:

- `startRow`
- `rowCount`
- `datetime_beginning_ept=MM/DD/YYYY HH:mmtoMM/DD/YYYY HH:mm`
- `type=ZONE`

Archived compatibility notes:

- Do not send `fields` for this pull path.
- Do not send `pnode_name` for this pull path.
- Keep filtering authoritative on the client side after normalization.

## AEP zone filtering strategy

1. Request small windows first for schema confirmation.
2. Normalize defensively across possible column variants.
3. Filter rows by AEP from `pnode_name` and zone-like fields client-side.

## Backfill chunking

- Default chunk size: 7 days (conservative for non-member usage)
- Supports 30-day initial pull and one-year pull (`--one-year`)
- One-year pull is resumable: rerun the same command to continue from existing valid chunks
- Manifest is written under `data/cache/reports/pjm_backfill_manifest_AEP_2024.json`

## Rate limiting

- PJM non-member limit is 6 connections/minute.
- Project default enforces <=5 connections/minute for safety margin.
- Effective throttle is `max(12.0, 60 / connections_per_minute)` seconds between requests.
- HTTP 429 path applies cooldown >= 90 seconds before retry.

## Data quality report fields

Quality report includes row counts, expected/missing hours, duplicates, DST day lengths, distribution stats, negative/zero/extreme price counts, and observed pnode metadata.

## Outputs generated

- Raw local: `data/raw/pjm/da_hrl_lmps/`
- Normalized local cache: `data/cache/pjm/da_hrl_lmps/`
- Quality report: `data/cache/reports/`

## Why outputs are ignored by Git

Raw and derived PJM data are private, potentially licensed/sensitive, and are intentionally excluded from source control.
Generated reports/manifests also remain ignored.

## Next step

Build the real AEP panel from cached real LMP and run real TFT/DeepAR baseline training.
