"""Pydantic models for the live predictions API contract.

Used in two places:
  1. json_writer.py validates the dict before atomic-renaming live_predictions.json
  2. FastAPI app uses LivePredictions / AppConfig as response models
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class DirectionPrediction(BaseModel):
    quantiles:        Dict[str, float] = Field(default_factory=dict)
    threshold_probs:  Dict[str, float] = Field(default_factory=dict)


class HorizonEntry(BaseModel):
    horizon_h:    int
    target_time:  str
    long:         DirectionPrediction
    short:        DirectionPrediction


class HistoricalAnchor(BaseModel):
    anchor_time:   str
    anchor_price:  float
    horizons:      List[HorizonEntry]


class SymbolPrediction(BaseModel):
    symbol:               str
    current_price:        float
    current_time:         str
    horizons:             List[HorizonEntry]
    historical:           List[HistoricalAnchor] = Field(default_factory=list)
    hist_anchor_step_h:   int


class LivePredictions(BaseModel):
    schema_version:    int = SCHEMA_VERSION
    generated_at:      str
    model_checkpoint:  str
    predictions:       List[SymbolPrediction]


class AppConfig(BaseModel):
    """Returned by /api/config — single source of truth for the dashboard."""
    symbols:         List[str]
    horizons_hours:  List[int]
    quantiles:       List[float]
    thresholds_pct:  List[float]
    direction:       str           # "long" | "short" | "both"
