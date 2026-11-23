"""Signal ingestion: schema, point-in-time store, normalisation, track records, combination."""
from .schema import Signal, SignalStore, SignalValidationError, FAMILIES, KINDS
from .normalize import zscore, rank_gauss, panel_zscore, grinold_expected_return, horizon_scale, age_decay, store_to_scores
from .performance import (forward_returns, cross_sectional_ic, ic_summary, track_record_asof, known_ics, ewma_ic,
                          ic_decay, newey_west_t)
from .combine import combine, Combination, METHODS as COMBINATION_METHODS, ic_table
from . import reference, synthetic, adapters

__all__ = [n for n in dir() if not n.startswith("_")]
