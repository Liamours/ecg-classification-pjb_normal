"""Where the data lives. Roots and dataset aliases come from configs/paths.yml, so a folder rename changes one line there.

A path string may start with @: `@logs`, `@inferences/run1` (a root) or `@mac400-scan/scan_29/x.jpg` (a dataset alias). Other strings stay plain paths.
"""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "paths.yml"


def _load() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def root(name: str) -> Path:
    return (REPO / _load()["roots"][name]).resolve()


def dataset(alias: str) -> Path:
    """The folder of a dataset by alias; a name that is not an alias is taken as the folder name under datasets/."""
    return root("datasets") / _load()["datasets"].get(alias, alias)


def resolve(p: str | Path) -> Path:
    s = str(p).replace("\\", "/")
    if not s.startswith("@"):
        return Path(p)
    head, _, rest = s[1:].partition("/")
    base = root(head) if head in _load()["roots"] else dataset(head)
    return base / rest if rest else base
