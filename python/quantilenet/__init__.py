from .loader import QuantileFeatureLoader, QUANTILE_FEATURE_DIM, quantile_feature_dim, extended_position_dim

POSITION_BASE_DIM = 10  # is_long, is_short, unrealized_pnl, hold_6h, hold_48h, sin_hour, cos_hour, sin_week, cos_week, is_weekend

__all__ = [
    "QuantileFeatureLoader",
    "QUANTILE_FEATURE_DIM",
    "POSITION_BASE_DIM",
    "quantile_feature_dim",
    "extended_position_dim",
]
