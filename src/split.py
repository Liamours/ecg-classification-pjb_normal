"""Nested cross-validation folds for the labeled phone photos: study groups, iterative multi-label stratification.

Unit: the study group, one photo with byte-identical copies merged (no patient key exists). Stratification at group level by
iterative stratification (Sechidis et al. 2011, `iterstrat`) on every diagnosis as written, a NORMAL column and the
collection batch, so each fold keeps each label's and each batch's share. Per repeat (one seed each): `outer_folds` outer folds,
each tested once; inside each outer training part a stratified `val_share` validation split. Writes the long-format manifest
`out` (relative_path, study_id, repeat, fold, role in train, val or test) and the per-fold counts `counts`, and checks that
no study group falls in two roles of one repeat and fold.

Usage:
    uv run python -m src.split --config configs/split.yml
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from src import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        pages = [r for r in csv.DictReader(fh) if r["page_style"] == cfg["page_style"]]
    with paths.resolve(cfg["duplicates"]).open(encoding="utf-8") as fh:
        md5 = {r["relative_path"]: r["md5"] for r in csv.DictReader(fh)}
    group_of = {}
    for r in pages:
        group_of[r["relative_path"]] = "s" + md5.get(r["relative_path"], r["relative_path"])[:12]
    groups = sorted(set(group_of.values()))
    gi = {g: i for i, g in enumerate(groups)}
    diagnoses = sorted({d for r in pages for d in json.loads(r[cfg["detailed_column"]] or "[]")} - {"NORMAL"})
    batches = sorted(set(cfg["batches"].values()))
    columns = ["NORMAL"] + diagnoses + [f"batch {b}" for b in batches]
    y = np.zeros((len(groups), len(columns)), int)
    for r in pages:
        i = gi[group_of[r["relative_path"]]]
        ds = set(json.loads(r[cfg["detailed_column"]] or "[]"))
        y[i, 0] |= r[cfg["label_column"]] == "NORMAL"
        for d in ds - {"NORMAL"}:
            y[i, 1 + diagnoses.index(d)] = 1
        y[i, 1 + len(diagnoses) + batches.index(cfg["batches"][r["relative_path"].split("/")[0]])] = 1
    print(f"{len(pages)} pages, {len(groups)} study groups, {len(columns)} stratification columns")
    rows, counts = [], []
    for rep, seed in enumerate(cfg["seeds"], 1):
        outer = MultilabelStratifiedKFold(n_splits=cfg["outer_folds"], shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(groups)), y), 1):
            inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=cfg["val_share"], random_state=seed)
            tr, va = next(inner.split(np.zeros(len(train_idx)), y[train_idx]))
            role = {groups[i]: "test" for i in test_idx} | {groups[train_idx[i]]: "train" for i in tr} | {groups[train_idx[i]]: "val" for i in va}
            assert len(role) == len(groups), "a study group has two roles"
            for r in pages:
                rows.append({"relative_path": r["relative_path"], "study_id": group_of[r["relative_path"]], "repeat": rep, "fold": fold, "role": role[group_of[r["relative_path"]]]})
            for part in ("train", "val", "test"):
                idx = [gi[g] for g, v in role.items() if v == part]
                counts.append({"repeat": rep, "fold": fold, "role": part, "groups": len(idx), **{c: int(y[idx, j].sum()) for j, c in enumerate(columns)}})
    for path, data in ((paths.resolve(cfg["out"]), rows), (paths.resolve(cfg["counts"]), counts)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    test = [c for c in counts if c["role"] == "test"]
    show = ["groups", "NORMAL", "ASD", "VSD", "PDA", "PS", "TOF", "PH", "DORV", "AVSD", "TGA", "COA"] + [f"batch {b}" for b in batches]
    print("test folds, groups per column, min to max over 15 folds:")
    for c in show:
        v = [t[c] for t in test]
        print(f"  {c}: {min(v)} to {max(v)} (total {sum(v) // len(cfg['seeds'])})")
    print(f"wrote {len(rows)} rows to {paths.resolve(cfg['out'])}")


if __name__ == "__main__":
    main()
