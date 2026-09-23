"""The data carriers default none of their fields.

A defaulted field cannot fail: a construction that leaves it out gets the default, silently, and
the default names the wrong thing. `Block.building_tier` defaulting to `SpacingDiscs` is how a
footprint region rebuilt without its tier was re-modelled as discs; `building_geometries`
defaulting to an empty frame gave any Block rebuilt without it no buildings, so every road read as
free. With no defaults, mypy lists every construction that leaves a field out.

The list is EXPLICIT, so a carrier added later is not covered until it is named here.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from reblock.compare import LensPrefixes, MethodCurve
from reblock.contracts import Block, Metrics, Proposal, Region, Result
from reblock.data.pools import Pools, PoolSpec
from reblock.methods.desire_lines import DesireField, WeightedLines
from reblock.pipeline import PipelineSpec, RunOutput
from reblock.transplant.donors import DonorFit, Donors
from reblock.transplant.gw import GWParams
from reblock.transplant.signature import SignatureParams
from reblock.transplant.transport import Transport, TransportParams

CARRIERS: list[type[Any]] = [
    Block, Proposal, Region, Metrics, Result, MethodCurve, RunOutput, PipelineSpec,
    PoolSpec, Pools, GWParams, TransportParams, Transport, SignatureParams, Donors, DonorFit,
    WeightedLines, DesireField, LensPrefixes]

# `attrs` is an open bag of per-source extras (`kblock_k`, a probe's `interior_boundaries`) that
# most constructions have nothing to put in, where an empty mapping is what "no extras" means. It
# is due a typed redesign of its own, which is where its default goes.
EXEMPT = {(Block, "attrs"), (Region, "attrs")}


def _defaulted(carrier: type[Any]) -> list[str]:
    return [f.name for f in dataclasses.fields(carrier)
            if f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING]


@pytest.mark.parametrize("carrier", CARRIERS, ids=lambda c: c.__name__)
def test_carrier_fields_have_no_defaults(carrier: type[Any]) -> None:
    unexempt = [name for name in _defaulted(carrier) if (carrier, name) not in EXEMPT]
    assert unexempt == [], f"{carrier.__name__} defaults {unexempt}"


def test_every_exemption_names_a_field_that_still_defaults() -> None:
    # Otherwise an exemption outlives its reason and would quietly cover a field reintroduced
    # under the same name.
    for carrier, name in EXEMPT:
        assert name in _defaulted(carrier), f"{carrier.__name__}.{name} is exempt for nothing"
