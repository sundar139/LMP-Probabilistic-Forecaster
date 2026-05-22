# LMP Probabilistic Forecaster

A local-first, production-style probabilistic forecasting platform for PJM day-ahead hourly zonal LMPs.

## Current implementation status

Implemented and committed through:
- foundation bootstrap
- smoke ingestion adapters
- single-zone panel builder
- single-zone baseline training
- real PJM ingestion support (this step)

This step adds real PJM AEP ingestion + validation support only.
No model training is performed here.

## Real PJM ingestion support

Automated real PJM API ingestion requires `PJM_API_KEY` configured in `.env`.

References:
- Feed definition: https://dataminer2.pjm.com/feed/da_hrl_lmps/definition
- API base: https://api.pjm.com/api/v1/
- Getting started guide:
  https://www.pjm.com/-/media/DotCom/etools/data-miner-2/data-miner-2-getting-started-guide.pdf

### Required env keys

Add these to `.env`:
- `PJM_API_KEY=`
- `PJM_API_BASE_URL=https://api.pjm.com/api/v1`
- `PJM_DATA_MINER_BASE_URL=https://dataminer2.pjm.com`

## Dry-run vs write commands

Dry probe:

```powershell
uv run python -m lmp_forecaster.cli inspect-pjm-api --zone AEP --start-date 2024-01-01 --end-date 2024-01-02
```

Dry backfill plan:

```powershell
uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --start-date 2024-01-01 --end-date 2024-01-31
```

Write 30-day pull:

```powershell
uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --start-date 2024-01-01 --end-date 2024-01-31 --write
```

Write one-year pull (resumable):

```powershell
uv run python -m lmp_forecaster.cli pull-real-pjm-lmp --zone AEP --one-year --write
```

Workflow recommendation: run 30-day first, then one-year. If interrupted, rerun the same one-year command to resume.

## Data privacy policy reminder

Raw PJM and all derived outputs are private local artifacts and are ignored by Git:
- `data/raw/**`
- `data/cache/**`
- `data/processed/**`
- `artifacts/`
- `lightning_logs/`
- `mlruns/`

## Troubleshooting

- Missing `PJM_API_KEY`:
  - set key in `.env`, rerun with `--write`
- 401/403 failures:
  - verify key validity/subscription scope
- PJM schema changes:
  - update normalization mapping in `src/lmp_forecaster/data/pjm_api.py`
- Rate limits:
  - reduce request cadence / chunk size
  - non-member users should stay <=6/min; this project defaults to 5/min
- Empty AEP results:
  - verify date range and filter behavior
- DST row differences:
  - expected for spring/fall transitions; inspect quality report
- Interrupted one-year pull:
  - rerun the same command to resume from valid chunks

## Quality gates

```powershell
uv run ruff check .
uv run mypy src
uv run pytest -q
```

## Next step

Build real AEP panel from cached real PJM LMP + weather and run real TFT/DeepAR baseline training.
