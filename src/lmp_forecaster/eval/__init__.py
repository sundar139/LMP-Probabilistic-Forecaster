"""Evaluation namespace."""

from lmp_forecaster.eval.calibration import (
    CalibrationDiagnosticConfig,
    classify_calibration_status,
    compute_interval_coverage_by_fold,
    compute_interval_coverage_by_horizon,
    compute_interval_width_by_horizon,
    compute_median_bias,
    compute_pinball_by_quantile,
    compute_quantile_crossing_rate,
    discover_latest_backtest_outputs,
    load_calibration_config,
    summarize_calibration_diagnostics,
    write_calibration_report,
)

__all__ = [
    "CalibrationDiagnosticConfig",
    "classify_calibration_status",
    "compute_interval_coverage_by_fold",
    "compute_interval_coverage_by_horizon",
    "compute_interval_width_by_horizon",
    "compute_median_bias",
    "compute_pinball_by_quantile",
    "compute_quantile_crossing_rate",
    "discover_latest_backtest_outputs",
    "load_calibration_config",
    "summarize_calibration_diagnostics",
    "write_calibration_report",
]
