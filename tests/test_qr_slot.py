"""hooks/qr_slot.py: every built page carries its QR-code slot, naming its own canonical URL."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qr_slot", ROOT / "hooks" / "qr_slot.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _hook()
URL = "https://geoffchurch.github.io/REU_Slum_Optimization/demo/"


def _page(url: str, slot: bool = True) -> str:
    """A page as overrides/main.html renders one: its canonical link and, after the sidebars, the
    slot. Read from the template itself, so a changed slot fails here too."""
    tpl = (ROOT / "overrides" / "main.html").read_text(encoding="utf-8")
    m = re.search(r'(<div class="sbu-qr" data-sbu-qr=)"\{\{ page.canonical_url \}\}"( hidden>)',
                  tpl)
    # As built pages carry it: glightbox re-serializes `hidden` as `hidden=""`.
    assert m is not None, "overrides/main.html's slot is not the one the audit looks for"
    return (f'<link rel="canonical" href="{url}">'
            + (f'{m.group(1)}"{url}" hidden=""></div>' if slot else ""))


def test_the_site_runs_the_audit() -> None:
    yml = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^hooks:\n(  - hooks/\S+[^\n]*\n)*  - hooks/qr_slot\.py\b", yml, flags=re.M)


def test_every_page_has_its_own_slot_and_the_404_none(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "index.html").write_text(_page(URL), encoding="utf-8")
    (tmp_path / "404.html").write_text("<html></html>", encoding="utf-8")
    assert hook.audit(tmp_path) == []
    (tmp_path / "demo" / "index.html").write_text(_page(URL, slot=False), encoding="utf-8")
    assert hook.audit(tmp_path) == [f"demo/index.html: QR slots [], not one naming {URL}"]
    (tmp_path / "demo" / "index.html").write_text(
        _page(URL).replace(f'data-sbu-qr="{URL}"', 'data-sbu-qr="https://example.org/"'),
        encoding="utf-8")
    assert len(hook.audit(tmp_path)) == 1
    (tmp_path / "404.html").write_text(_page(URL), encoding="utf-8")
    assert any(p.startswith("404.html") for p in hook.audit(tmp_path))
