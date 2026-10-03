"""The record assembler on small hand-made digitize outputs: no dataset or model needed."""
import csv
import json

import numpy as np

from src.records import page_record, source_key

CFG = {"fs": 500, "duration_s": 4.0, "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]}


def write_panel(page, index, names, flags, seconds=3.0, error=""):
    (page / f"panel{index}.json").write_text(json.dumps({"error": error}), encoding="utf-8")
    t = np.arange(0, seconds, 0.004)
    with (page / f"panel{index}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s"] + [f"{n}_mV" for n in names] + [f"{n}_flag" for n in names])
        for k in range(len(t)):
            w.writerow([f"{t[k]:.4f}"] + [f"{np.sin(2 * np.pi * t[k] * (i + 1)):.4f}" for i in range(len(names))] + flags)


def test_leads_land_on_one_time_base_and_the_better_duplicate_wins(tmp_path):
    write_panel(tmp_path, 0, ["I", "II", "III"], ["ok", "low_quality", "ok"])
    write_panel(tmp_path, 1, ["I", "II", "III"], ["low_quality", "ok", "ok"])
    write_panel(tmp_path, 2, ["V1", "V2", "V3"], ["ok", "ok", "ok"], error="ValueError: grid")
    rec = page_record(tmp_path, CFG)
    assert rec["signal"].shape == (12, 2000)
    assert rec["present"][:3].all() and rec["ok"][:3].all() and not rec["present"][6:9].any()
    assert np.isfinite(rec["signal"][0, :1400]).all() and np.isnan(rec["signal"][0, 1600:]).all()  # a 3 s lead in a 4 s window
    assert abs(float(np.nanmax(rec["signal"][0])) - 1.0) < 0.05


def test_source_key_reads_windows_and_posix_paths():
    assert source_key("..\\..\\datasets\\ecg-mac400-phone_photo\\NORMAL\\NORMAL 1.jpg") == ("ecg-mac400-phone_photo", "NORMAL/NORMAL 1.jpg")
    assert source_key("../../datasets/ecg-mac400-scan/scan_29/a.jpg") == ("ecg-mac400-scan", "scan_29/a.jpg")
