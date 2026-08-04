"""Malaria selection metrics, declared into the ``idd_tools`` metric registry.

The scam / gam / lm worker computes these per spec — in-sample and across the temporal OOS
windows — and ``finalize`` writes them to ``selection_summary.parquet`` with per-window
columns named ``<window>__<metric>``. ``idd_tools`` does not compute them, so each registers
with ``fn=None``: that is enough to be *selected* (by ``sample`` / ``space`` / ``family``)
and *oriented* (higher / lower), which is all the ranking machinery needs.

This is the metric half of "setting up a spec design": :mod:`malaria_spec_design` declares
the typed covariate space; this module declares its metric vocabulary. Import it (or call
:func:`register_malaria_metrics`) once, then drive selection off the registry:

    from idd_forecast_mbp.select.malaria_metrics import register_malaria_metrics, windowed
    from idd_tools.model_selection import select_metrics, Sample
    register_malaria_metrics()
    specs   = select_metrics(sample=Sample.OUT_OF_SAMPLE, space="pfpr")   # the 3 PfPR OOS metrics
    metrics = windowed(specs, WINDOWS)                                    # {<window>__<metric>: dir}
"""

from __future__ import annotations

from collections.abc import Iterable

from idd_tools.model_selection import (
    METRIC_REGISTRY,
    Direction,
    MetricFamily,
    MetricSpec,
    Sample,
    metric_directions,
    register_metric,
)

# name == the worker's metric-column suffix (what finalize prefixes with "<window>__").
# tags["space"]: "pfpr" = natural PfPR units (what the report ranks on) / "logit" = model space.
_METRICS: tuple[MetricSpec, ...] = (
    # --- PfPR space -----------------------------------------------------------------------
    MetricSpec("oos_pfpr_rmse", "PfPR RMSE (OOS)", Direction.LOWER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "pfpr"}),
    MetricSpec("oos_pfpr_mae", "PfPR MAE (OOS)", Direction.LOWER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "pfpr"}),
    MetricSpec("oos_pfpr_r", "PfPR corr (OOS)", Direction.HIGHER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "pfpr"}),
    # --- logit (modeling) space -----------------------------------------------------------
    MetricSpec("oos_rmse", "logit RMSE (OOS)", Direction.LOWER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "logit"}),
    MetricSpec("oos_mae", "logit MAE (OOS)", Direction.LOWER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "logit"}),
    MetricSpec("oos_r_sq", "logit R² (OOS)", Direction.HIGHER, Sample.OUT_OF_SAMPLE,
               MetricFamily.ACCURACY, tags={"space": "logit"}),
)


def register_malaria_metrics() -> None:
    """Register the malaria selection metrics into the shared registry (idempotent)."""
    for spec in _METRICS:
        if spec.name not in METRIC_REGISTRY:
            register_metric(spec)


def windowed(
    specs: Iterable[MetricSpec | str], windows: Iterable[str], *, sep: str = "__"
) -> dict[str, str]:
    """Cross selected metrics with the run's OOS windows into the ranker's criteria dict.

    Returns ``{f"{window}{sep}{metric}": direction}`` — one criterion per (window, metric),
    matching ``finalize``'s per-window column naming. This is how "all the windows" enter
    the ranking (each as its own criterion), rather than being averaged into one number.
    """
    base = metric_directions(specs)  # {metric_name: "higher"/"lower"}
    return {f"{w}{sep}{name}": direction for w in windows for name, direction in base.items()}


register_malaria_metrics()  # declare on import
