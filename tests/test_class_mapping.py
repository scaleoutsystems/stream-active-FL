"""Tests for stream_active_fl.core.build_class_mapping."""

from __future__ import annotations

import pytest

from stream_active_fl.core import (
    CATEGORY_NAME_TO_ID,
    NUM_CLASSES,
    build_class_mapping,
)


def test_default_uses_all_zod_classes():
    cm = build_class_mapping(None)
    assert cm.names == tuple(CATEGORY_NAME_TO_ID)
    # NUM_CLASSES already includes the +1 background slot.
    assert cm.num_classes == NUM_CLASSES


def test_subset_remaps_to_contiguous_zero_indexed_ids():
    cm = build_class_mapping(["Pedestrian", "Vehicle", "TrafficSign"])
    assert cm.names == ("Pedestrian", "Vehicle", "TrafficSign")
    assert cm.name_to_id == {"Pedestrian": 0, "Vehicle": 1, "TrafficSign": 2}
    assert cm.id_to_name == {0: "Pedestrian", 1: "Vehicle", 2: "TrafficSign"}
    # Model labels are 1-indexed because label 0 is reserved for background.
    assert cm.label_to_name == {1: "Pedestrian", 2: "Vehicle", 3: "TrafficSign"}
    # +1 for background.
    assert cm.num_classes == 4


def test_single_class_subset():
    cm = build_class_mapping(["Vehicle"])
    assert cm.names == ("Vehicle",)
    assert cm.name_to_id == {"Vehicle": 0}
    assert cm.label_to_name == {1: "Vehicle"}
    assert cm.num_classes == 2  # background + Vehicle


def test_unknown_class_rejected():
    with pytest.raises(ValueError, match="Unknown class"):
        build_class_mapping(["NotARealClass"])


def test_subset_preserves_user_supplied_order():
    """Order matters: it controls model output channel layout."""
    cm = build_class_mapping(["TrafficSign", "Pedestrian"])
    assert cm.names == ("TrafficSign", "Pedestrian")
    assert cm.name_to_id == {"TrafficSign": 0, "Pedestrian": 1}
