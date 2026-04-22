"""Stream-block labeling for per-domain evaluation.

Maps validation frames to the same block labels the streaming manifest
uses for ordering training data (e.g. city_day_cloudy, city_night), so
per-domain mAP can be reported at the joint granularity of the manifest
rather than only the marginal axes (time_of_day, road_condition, road_type).

Currently supports the cityday_curated ordering; unknown strategies fall
back to no stream_block labels (callers should pass extended_domain_dims
without stream_block in that case, or omit the attach step).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


STREAM_BLOCK_DIM: str = "stream_block"


def _weather_bucket(scraped_weather: Optional[str], road_condition: Optional[str]) -> str:
    """Coarse weather bucket from raw ZOD fields.

    Mirrors tools/preprocessing/build_manifests.py::_weather_bucket so
    validation frames get the same bucket a train frame would receive.
    """
    w = (scraped_weather or "").lower()
    rc = (road_condition or "").lower()
    if "snow" in w or "snow" in rc:
        return "snow"
    if "rain" in w or "wet" in rc:
        return "rain_wet"
    if "fog" in w:
        return "fog"
    if "cloud" in w or "overcast" in w:
        return "cloudy"
    return "clear"


def cityday_curated_block_label(meta: Mapping[str, Any]) -> str:
    """Map one frame's metadata to a cityday_curated block label.

    Folding rule matches the manifest builder: fog is merged into cloudy
    within the city_day weather buckets.  Unknown combinations return
    'other'.
    """
    rt = str(meta.get("road_type") or "unknown")
    tod = str(meta.get("time_of_day") or "unknown")
    wb = _weather_bucket(meta.get("scraped_weather"), meta.get("road_condition"))
    if wb == "fog":
        wb = "cloudy"
    if rt == "city":
        if tod == "day":
            return f"city_day_{wb}"
        if tod == "twilight":
            return "city_twilight"
        return "city_night"
    if rt == "arterial-urban":
        return "arterial-urban_day" if tod == "day" else "arterial-urban_twi-night"
    if rt == "highway":
        return "highway_day" if tod == "day" else "highway_twi-night"
    if rt == "arterial-rural":
        return "arterial-rural_day" if tod == "day" else "arterial-rural_twi-night"
    if rt == "smaller-rural":
        return "smaller-rural_all"
    return "other"


# Registry of strategy name -> per-frame labeler.  Extend here when new
# manifest ordering strategies are added.
_STRATEGY_TO_LABELER = {
    "cityday_curated_blocks": cityday_curated_block_label,
}


def get_block_labeler(strategy: Optional[str]):
    """Return the per-frame labeler for a manifest ordering strategy.

    Returns None when the strategy is unknown or missing, so callers can
    skip attaching stream_block labels rather than emit bogus ones.
    """
    if strategy is None:
        return None
    return _STRATEGY_TO_LABELER.get(str(strategy))


def attach_stream_blocks(
    domain_labels: Dict[str, Dict[str, Any]],
    frames: List[Dict[str, Any]],
    strategy: Optional[str],
) -> bool:
    """Enrich a frame_id -> metadata mapping with a stream_block field.

    Args:
        domain_labels: Mapping from frame_id to metadata dict (mutated in
            place).  Any frame_id missing from domain_labels is ignored.
        frames: List of manifest frame entries used to read the raw
            metadata fields (road_type, time_of_day, scraped_weather,
            road_condition).  Typically val_stream.frames or a superset.
        strategy: Manifest ordering strategy (e.g. cityday_curated_blocks).

    Returns:
        True if stream_block labels were attached, False if the strategy
        is unknown and no labels were written.
    """
    labeler = get_block_labeler(strategy)
    if labeler is None:
        return False
    for frame in frames:
        fid = str(frame.get("frame_id"))
        if fid not in domain_labels:
            continue
        domain_labels[fid][STREAM_BLOCK_DIM] = labeler(frame)
    return True
