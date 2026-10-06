"""Decision rule for the hierarchical labels: one threshold per level (group, diagnosis), chosen on validation pages by micro F1,
and a diagnosis kept only when its group is predicted. A shared threshold per level uses every positive of that level, which a
per-label threshold on a small validation split cannot (most diagnoses have 0 to 4 validation pages)."""
import numpy as np
import yaml
from sklearn.metrics import f1_score

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def parents(labels: list[str], groups_config, merge: dict, unknown: str) -> np.ndarray:
    """Index of each diagnosis label's group label (-1 for group labels)."""
    groups = yaml.safe_load(groups_config.read_text(encoding="utf-8"))["groups"]
    group_of = {f"diagnosis: {merge.get(d, d)}": f"group: {g}" for d, g in groups.items()} | {f"diagnosis: {unknown}": f"group: {unknown}"}
    return np.array([labels.index(group_of[l]) if l.startswith("diagnosis") else -1 for l in labels])


def predict(prob: np.ndarray, thresholds: dict, parent: np.ndarray) -> np.ndarray:
    is_group = parent < 0
    pred = np.zeros(prob.shape, bool)
    pred[:, is_group] = prob[:, is_group] >= thresholds["group"]
    pred[:, ~is_group] = (prob[:, ~is_group] >= thresholds["diagnosis"]) & pred[:, parent[~is_group]]
    return pred


def choose(prob: np.ndarray, y: np.ndarray, parent: np.ndarray) -> dict:
    """Group threshold by group micro F1 on validation, then diagnosis threshold by diagnosis micro F1 under the group rule."""
    is_group = parent < 0
    t = {"group": 0.5, "diagnosis": 0.5}
    t["group"] = float(max(GRID, key=lambda g: f1_score(y[:, is_group], prob[:, is_group] >= g, average="micro", zero_division=0)))
    t["diagnosis"] = float(max(GRID, key=lambda d: f1_score(y[:, ~is_group], predict(prob, {"group": t["group"], "diagnosis": d}, parent)[:, ~is_group], average="micro", zero_division=0)))
    return t
