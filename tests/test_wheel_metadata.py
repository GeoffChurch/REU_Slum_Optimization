"""The reblock wheel declares no dependencies.

The DrawRoad widget's Pyodide runtime installs this wheel with `micropip.install(wheelUrl)`
(web/src/py/runtime.ts), and micropip resolves every `Requires-Dist` the wheel declares -- from
PyPI, which has no wasm wheels for geopandas and the rest of the science stack. So what reblock
needs to run lives in pyproject.toml's `runtime` dependency group, which uv installs and no build
writes into the wheel; moving any of it into `[project] dependencies` would break the published
widget while every Python test stayed green. This reads the wheel `make wheel` built (`test-py`
depends on it), the one the site ships.
"""
from __future__ import annotations

import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_built_wheel_declares_no_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    name, version = project["name"], project["version"]
    wheel = ROOT / "dist" / f"{name}-{version}-py3-none-any.whl"
    assert wheel.exists(), (
        f"{wheel} is missing -- `make wheel` builds it, and `test-py` depends on that target, so "
        f"reaching this line without it means the dependency stopped firing")
    with zipfile.ZipFile(wheel) as zf:
        metadata = zf.read(f"{name}-{version}.dist-info/METADATA").decode()
    requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
    assert requires == [], f"the wheel would make micropip resolve {requires} from PyPI"
