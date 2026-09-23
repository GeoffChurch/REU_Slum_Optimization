"""The shipped permeability parameters, for tests that score with them.

`PermeabilityParams` has no defaults: its one source is conf/permeability.yaml, read by
`load_permeability_config`. A test that varies one parameter starts from `SHIPPED` and `replace`s
that field, so every other field is the shipped value rather than an invented one. Shared through a
non-test module, the precedent tests/scoring_fixtures.py set.
"""
from __future__ import annotations

from reblock.compare import load_permeability_config

SHIPPED = load_permeability_config().params
