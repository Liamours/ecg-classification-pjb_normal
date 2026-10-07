"""Explanations of the final ECGFounder model: which digitized samples of which lead drive its output.

Methods (`methods` in the config): saliency, gradient times input, Integrated Gradients, SmoothGrad, GradientSHAP, DeepLIFT,
Grad-CAM (Captum) and occlusion of 0.2 s stretches (own code, in record space). The model reads each lead's digitized stretch
repeated to 10 s (`tile`), so an input attribution is folded back onto the record: every input sample adds to the record sample it
copies. Grad-CAM has no lead resolution (its layer mixes the leads) and gives every lead the same time profile. The explained output
is the group label with the highest probability: it decides NORMAL (no group above its threshold) against PJB.

Test pages (`split`): per page every method's record attribution, the probability drop when a whole lead is zeroed, deletion and
insertion curves (Petsiuk et al. 2018; most attributed samples first, plus a random order) and, per true label, Integrated Gradients'
share of attribution per lead. Example pages (`pages`): re-digitized with every per-panel file kept (`digitizer`), so each digitized
sample maps to its photo pixel through the panel's grid map; then the same attributions on that fresh record.

Writes to `out_dir`: `test/<page>.npz`, `pages/<page>/` (digitizer output and `xai.npz`); to `report_dir`: `faithfulness.csv`
(per page, method and fraction), `faithfulness_summary.csv` (area under the deletion and insertion curves per method),
`lead_map.csv` (per label, mean share per lead), `viewer.json` (the data of `src.xai_viewer`). One progress line with ETA per part,
a log in `log_dir`; finished pages are skipped on a rerun.

Usage:
    uv run python -m src.xai --config configs/xai.yml
"""
import argparse
import csv
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter1d
from captum.attr import DeepLift, GradientShap, InputXGradient, IntegratedGradients, LayerAttribution, LayerGradCam, NoiseTunnel, Saliency
from PIL import Image, ImageOps

from src import explain as explain_step
from src import paths
from src.explain import tile_index
from src.linear_probe import label_matrix, load_folds, write_csv
from src.models import LEADS
from src.progress import Progress

REPO = Path(__file__).resolve().parents[1]
MM_PER_S = 25.0


def load_model(cfg: dict, device: str):
    return explain_step.load_model(cfg["model"], device)


def read_record(page_dir: Path) -> np.ndarray:
    return np.genfromtxt(page_dir / "record.csv", delimiter=",", skip_header=1)[:, 1:].T


def fold(a: np.ndarray, idx: list[np.ndarray], n_rec: int) -> np.ndarray:
    r = np.full((len(idx), n_rec), np.nan)
    for l, ix in enumerate(idx):
        if len(ix):
            r[l, np.unique(ix)] = 0.0
            np.add.at(r[l], ix, a[l])
    return r


def probs(net, xs: torch.Tensor, target: int, bs: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            out.append(torch.sigmoid(net(xs[i:i + bs])[:, target]).cpu().numpy())
    return np.concatenate(out)


def attribute(net, x: torch.Tensor, base: torch.Tensor, target: int, name: str, p: dict, bs: int) -> np.ndarray:
    """Input attribution (leads x input samples) of one method for one page and one output, against the input `base`."""
    x = x.clone().requires_grad_(True)
    noise = p.get("noise", 0) * float(x.max() - x.min())
    if name == "saliency":
        a = Saliency(net).attribute(x, target=target, abs=True)
    elif name == "input_x_gradient":
        a = InputXGradient(net).attribute(x, target=target)
    elif name == "integrated_gradients":
        a = IntegratedGradients(net).attribute(x, baselines=base, target=target, n_steps=p["steps"], internal_batch_size=bs)
    elif name == "smoothgrad":
        a = NoiseTunnel(Saliency(net)).attribute(x, nt_type="smoothgrad", nt_samples=p["samples"], nt_samples_batch_size=bs, stdevs=noise, target=target, abs=False)
    elif name == "gradient_shap":
        a = GradientShap(net).attribute(x, baselines=base, n_samples=p["samples"], stdevs=noise, target=target)
    elif name == "deeplift":
        a = DeepLift(net).attribute(x, baselines=base, target=target)
    elif name == "grad_cam":
        layer = net.backbone.get_submodule(p["layer"])
        cam = LayerGradCam(net, layer).attribute(x, target=target, relu_attributions=True)
        a = LayerAttribution.interpolate(cam, (x.shape[-1],), "linear").expand_as(x)
    else:
        raise ValueError(f"unknown method {name}")
    return a.detach()[0].float().cpu().numpy()


def occlusion(net, x: torch.Tensor, base: torch.Tensor, idx: list[np.ndarray], target: int, window: int, n_rec: int, bs: int) -> tuple[np.ndarray, np.ndarray]:
    """Record attribution from replacing `window` record samples of one lead at a time by the baseline (probability drop spread
    over the stretch), and the drop when each whole lead is replaced (`src.explain.occlude`, the pipeline's own step)."""
    blocks, lead_drop, _ = explain_step.occlude(explain_step.TorchScorer(net, x.device, bs), x[0].cpu().numpy(), base[0].cpu().numpy(), idx, target, window)
    r = np.full((len(idx), n_rec), np.nan)
    r[:, :blocks.shape[1]] = blocks
    return r, lead_drop


def curves(net, x: torch.Tensor, base: torch.Tensor, idx: list[np.ndarray], r: np.ndarray, target: int, fractions: list[float], bs: int) -> tuple[np.ndarray, np.ndarray]:
    """Target probability as the most attributed record samples are replaced by the baseline (deletion) or restored onto the
    baseline (insertion), at each fraction of the digitized samples."""
    lead, sample = np.nonzero(np.isfinite(r))
    order = np.argsort(-r[lead, sample], kind="stable")
    rank = np.full(r.shape, np.inf)
    rank[lead[order], sample[order]] = np.arange(len(order))
    rank_in = torch.from_numpy(np.stack([rank[l, ix] if len(ix) else np.full(x.shape[-1], np.inf) for l, ix in enumerate(idx)])).to(x.device)
    masks = [(rank_in < round(f * len(order)))[None].float() for f in fractions]
    p = probs(net, torch.cat([x * (1 - m) + base * m for m in masks] + [base * (1 - m) + x * m for m in masks]), target, bs)
    return p[:len(fractions)], p[len(fractions):]


def baseline(x: torch.Tensor, kind: str, p: dict, fs: int) -> torch.Tensor:
    """What a removed sample becomes: `zero` (the all-zero input) or `blur` (the input smoothed by a Gaussian of `sigma_s`,
    the blurred baseline of Sturmfels et al. 2020)."""
    if kind == "zero":
        return torch.zeros_like(x)
    return torch.from_numpy(gaussian_filter1d(x.cpu().numpy(), p["sigma_s"] * fs, axis=-1, mode="nearest")).to(x.device)


def explain(net, rec: np.ndarray, target: int, kind: str, cfg: dict, device: str) -> dict:
    """Every method's record attribution and the whole-lead drops for one record, against the baseline `kind`."""
    x = torch.from_numpy(net.adapter.prep(rec))[None].to(device)
    base = baseline(x, kind, cfg["baselines"][kind], cfg["fs"])
    idx = tile_index(rec, x.shape[-1])
    out = {}
    for name, p in cfg["methods"].items():
        if name == "occlusion":
            out[name], out["lead_drop"] = occlusion(net, x, base, idx, target, int(p["window_s"] * cfg["fs"]), rec.shape[1], cfg["batch_size"])
        else:
            out[name] = fold(attribute(net, x, base, target, name, p, cfg["batch_size"]), idx, rec.shape[1])
    with torch.no_grad():
        out["base_prob"] = float(torch.sigmoid(net(base))[0, target])
    return out | {"x": x, "base": base, "idx": idx}


def page_folder(rel: str) -> str:
    p = Path(rel)
    return f"{p.parent.name}__{p.stem}".replace(" ", "_")


def test_part(net, labels, thresholds, parent, tc, cfg, kind, device, log) -> None:
    out = paths.resolve(cfg["out_dir"]) / kind / "test"
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(paths.resolve(cfg["inputs"]), allow_pickle=False)
    pages = list(d["relative_path"])
    lab_cfg = {k: tc[k] for k in ("dataset", "groups_config", "label_column", "detailed_column", "merge", "unknown_label", "folds")}
    tc_labels, y = label_matrix(lab_cfg, pages)
    assert tc_labels == labels, "label order differs from the model's"
    role = load_folds(lab_cfg, pages)[(cfg["split"]["repeat"], cfg["split"]["fold"])]
    test = [i for i in np.flatnonzero(role == "test") if not (out / f"{page_folder(pages[i])}.npz").exists()]
    run_dir, groups = paths.resolve(cfg["run"]), np.flatnonzero(parent < 0)
    rng = np.random.default_rng(cfg["faithfulness"]["seed"])
    bar = Progress(len(test), f"xai test pages, {kind} baseline", "pages", log)
    for i in test:
        rec = read_record(run_dir / page_folder(pages[i]))
        if np.isnan(rec).all():  # the digitizer read no lead of this page: nothing to attribute
            np.savez_compressed(out / f"{page_folder(pages[i])}.npz", skipped=1)
            bar.step(1, f"{pages[i]} (no digitized lead)")
            continue
        x = torch.from_numpy(net.adapter.prep(rec))[None].to(device)
        assert np.allclose(x[0].cpu().numpy(), d["x"][i], atol=1e-4), f"{pages[i]}: record and cached input differ"
        with torch.no_grad():
            p = torch.sigmoid(net(x))[0].cpu().numpy()
        target = int(groups[np.argmax(p[groups])])
        e = explain(net, rec, target, kind, cfg, device)
        save = {"target": target, "prob": p, "truth": y[i], "lead_drop": e["lead_drop"], "base_prob": e["base_prob"]}
        fr = cfg["faithfulness"]["fractions"]
        for name in cfg["methods"]:
            save[f"attr_{name}"] = e[name].astype(np.float32)
            save[f"deletion_{name}"], save[f"insertion_{name}"] = curves(net, e["x"], e["base"], e["idx"], e[name], target, fr, cfg["batch_size"])
        first = e[next(iter(cfg["methods"]))]
        shuffled = np.where(np.isfinite(first), rng.random(first.shape), np.nan)
        save["deletion_random"], save["insertion_random"] = curves(net, e["x"], e["base"], e["idx"], shuffled, target, fr, cfg["batch_size"])
        lm = cfg["lead_map"]
        for j in np.flatnonzero(y[i]):
            a = np.abs(fold(attribute(net, e["x"], e["base"], int(j), lm["method"], cfg["methods"][lm["method"]], cfg["batch_size"]), e["idx"], rec.shape[1]))
            per_lead = np.nansum(a, axis=1)
            save[f"leadshare_{j}"] = per_lead / max(per_lead.sum(), 1e-12)
        np.savez_compressed(out / f"{page_folder(pages[i])}.npz", **save)
        bar.step(1, pages[i])
    bar.close()


def digitize(cfg: dict, files: list[Path], out: Path, log) -> None:
    dg = cfg["digitizer"]
    repo = (REPO / dg["repo"]).resolve()
    todo = [f for f in files if not (out / page_folder(f"{f.parent.name}/{f.name}") / "page.json").exists()]
    if not todo:
        return
    cmd = [str((REPO / dg["python"]).resolve()), "-m", "src.digitize.run", "--config", dg["config"], "--layout", dg["layout"], "--out-dir", str(out), "--device", dg["device"]] + [str(f) for f in todo]
    log.info("digitize %d pages: %s", len(todo), " ".join(cmd))
    subprocess.run(cmd, cwd=repo, env=os.environ | {"PYTHONPATH": str(repo)}, check=True)


def upright(source: Path, page: dict) -> np.ndarray:
    """The page as the digitizer saw it: EXIF orientation, scaled by its work scale, turned upright (RGB)."""
    img = np.asarray(ImageOps.exif_transpose(Image.open(source).convert("RGB")))
    s = page["work_scale"]
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return np.rot90(img, page["rotation_ccw_deg"] // 90).copy()


def pixel_map(page_dir: Path, fs: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per lead of the record: page pixels (upright frame) of its digitized samples and the record sample each one became
    (src.digitize.reverse.lead_pixels and assemble.page_record of the digitization repo)."""
    out = {}
    for lab in json.loads((page_dir / "record.json").read_text(encoding="utf-8"))["leads"]:
        if lab["status"] == "missing":
            continue
        pj = json.loads((page_dir / f"{lab['panel']}.json").read_text(encoding="utf-8"))
        g, co, box, lead = pj["grid"], np.array(pj["crop_origin"], float), pj["box"], pj["leads"][lab["lead"]]
        a = np.stack([np.array(g["a1"]), np.array(g["a2"])])
        to_mm = lambda pts: np.linalg.solve(a.T, (pts - np.array(g["origin"])).T).T
        left_mm = to_mm(np.array([[box[0] - co[0], (box[1] + box[3]) / 2 - co[1]]]))[0, 0]
        with (page_dir / f"{lab['panel']}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        t = np.array([float(r["time_s"]) for r in rows if r[f"{lab['lead']}_mV"]])
        mv = np.array([float(r[f"{lab['lead']}_mV"]) for r in rows if r[f"{lab['lead']}_mV"]])
        mm = np.column_stack([lead["x0_mm"] + t * MM_PER_S + lead["bin_mm"] / 2, lead["baseline_mm"] - mv * pj["gain_mm_per_mV"]])
        px = np.array(g["origin"]) + mm @ a + co
        start = (lead["x0_mm"] - left_mm) / MM_PER_S
        out[lab["lead"]] = (px, np.round((start + t) * fs).astype(int))
    return out


def pages_part(net, labels, thresholds, parent, cfg, kind, device, log) -> None:
    out = paths.resolve(cfg["out_dir"]) / "pages"
    out.mkdir(parents=True, exist_ok=True)
    files = [paths.dataset(cfg["dataset"]) / rel for rel in cfg["pages"].values()]
    digitize(cfg, files, out, log)
    groups = np.flatnonzero(parent < 0)
    todo = [rel for rel in cfg["pages"].values() if not (out / page_folder(rel) / f"xai_{kind}.npz").exists()]
    bar = Progress(len(todo), f"xai example pages, {kind} baseline", "pages", log)
    for rel in todo:
        d = out / page_folder(rel)
        rec = read_record(d)
        x = torch.from_numpy(net.adapter.prep(rec))[None].to(device)
        with torch.no_grad():
            p = torch.sigmoid(net(x))[0].cpu().numpy()
        target = int(groups[np.argmax(p[groups])])
        e = explain(net, rec, target, kind, cfg, device)
        save = {"target": target, "prob": p, "record": rec.astype(np.float32), "lead_drop": e["lead_drop"], "base_prob": e["base_prob"]} | {f"attr_{m}": e[m].astype(np.float32) for m in cfg["methods"]}
        for lead, (px, ri) in pixel_map(d, cfg["fs"]).items():
            save[f"px_{lead}"], save[f"ri_{lead}"] = px.astype(np.float32), ri
        np.savez_compressed(d / f"xai_{kind}.npz", **save)
        bar.step(1, rel)
    bar.close()


def summarize(labels, cfg, kind) -> None:
    """Faithfulness tables and the label-by-lead map from the test-page files of one baseline."""
    out, report = paths.resolve(cfg["out_dir"]) / kind / "test", paths.resolve(cfg["report_dir"]) / kind
    report.mkdir(parents=True, exist_ok=True)
    fr = np.array(cfg["faithfulness"]["fractions"])
    rows, auc, shares, gap = [], {}, {}, []
    for f in sorted(out.glob("*.npz")):
        z = np.load(f)
        if "skipped" in z.files:
            continue
        if "base_prob" in z.files:
            gap.append(float(z["base_prob"]) - float(z["prob"][int(z["target"])]))
        for m in list(cfg["methods"]) + ["random"]:
            dl, ins = z[f"deletion_{m}"], z[f"insertion_{m}"]
            rows += [[f.stem, m, f"{a:.2f}", f"{b:.5f}", f"{c:.5f}"] for a, b, c in zip(fr, dl, ins)]
            auc.setdefault(m, []).append((np.trapezoid(dl, fr) / dl[0] if dl[0] > 0 else np.nan, np.trapezoid(ins, fr) / dl[0] if dl[0] > 0 else np.nan))
        for k in z.files:
            if k.startswith("leadshare_"):
                shares.setdefault(int(k[10:]), []).append(z[k])
    write_csv(report / "faithfulness.csv", ["page", "method", "fraction", "deletion_prob", "insertion_prob"], rows)
    summary = [[m, len(v), f"{np.nanmean([a for a, _ in v]):.4f}", f"{np.nanmean([b for _, b in v]):.4f}"] for m, v in auc.items()]
    write_csv(report / "faithfulness_summary.csv", ["method", "pages", "deletion_auc", "insertion_auc"], summary)
    lm = [[labels[j], len(v)] + [f"{s:.4f}" for s in np.mean(v, axis=0)] for j, v in sorted(shares.items(), key=lambda kv: -len(kv[1])) if len(v) >= cfg["lead_map"]["min_pages"]]
    write_csv(report / "lead_map.csv", ["label", "pages"] + LEADS, lm)
    if gap:
        print(f"\n{kind} baseline: its probability minus the page's, median {np.median(gap):.4f} (quartiles {np.percentile(gap, 25):.4f} to {np.percentile(gap, 75):.4f})")
    print(f"| Method ({kind} baseline) | Pages | Deletion AUC (lower is better) | Insertion AUC (higher is better) |\n|---|---|---|---|")
    for r in sorted(summary, key=lambda r: float(r[2])):
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--part", choices=["all", "test", "pages", "summary"], default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"xai-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("xai")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, labels, thresholds, parent, tc = load_model(cfg, device)
    log.info("start part %s on %s, config %s", args.part, device, cfg)
    for kind in cfg["baselines"]:
        if args.part in ("all", "test"):
            test_part(net, labels, thresholds, parent, tc, cfg, kind, device, log)
        if args.part in ("all", "pages"):
            pages_part(net, labels, thresholds, parent, cfg, kind, device, log)
        if args.part in ("all", "summary"):
            summarize(labels, cfg, kind)


if __name__ == "__main__":
    main()
