# -*- coding: utf-8 -*-
# 修复诊断独立工具：
#   1) 读 cfu csv，独立于阈值算 PR-AUC / ROC-AUC（cfu_scoring 也打印，这里可单独跑）。
#   2) 读所有 repair_*.json，画 noise_recall vs tail_false_repair_rate trade-off 图。
# 用法:
#   python scripts/repair_diagnostics.py --cfu_csv logs/cfu_xxx.csv
#   python scripts/repair_diagnostics.py --repair_glob 'logs/repair_*.json'

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd


def auc_from_cfu(cfu_csv):
    from sklearn.metrics import roc_auc_score, average_precision_score
    df = pd.read_csv(cfu_csv)
    y = df["is_injected_noise"].values
    score = -df["CFU_delete"].values
    print(f"[{cfu_csv}] n={len(df)} noise_ratio={y.mean():.4f}")
    print(f"  ROC-AUC = {roc_auc_score(y, score):.4f}")
    print(f"  PR-AUC  = {average_precision_score(y, score):.4f}")


def tradeoff_plot(repair_glob):
    rows = []
    for p in sorted(glob.glob(repair_glob)):
        with open(p) as f:
            d = json.load(f)
        rows.append(d)
    if not rows:
        print("no repair json found")
        return
    df = pd.DataFrame(rows)
    print(df[["selection", "action", "repair_count", "noise_precision",
              "noise_recall", "tail_false_repair_rate", "head_false_repair_rate"]].to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["tail_false_repair_rate"], df["noise_recall"],
                   s=df["repair_count"] / df["repair_count"].max() * 200, alpha=0.6)
        for _, r in df.iterrows():
            ax.annotate(r["selection"], (r["tail_false_repair_rate"], r["noise_recall"]), fontsize=7)
        ax.set_xlabel("tail false repair rate")
        ax.set_ylabel("noise recall")
        ax.set_title("repair trade-off")
        out = "logs/repair_tradeoff.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"plot -> {out}")
    except Exception as e:
        print(f"plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", default=None)
    ap.add_argument("--repair_glob", default="logs/repair_*.json")
    args = ap.parse_args()
    if args.cfu_csv:
        auc_from_cfu(args.cfu_csv)
    tradeoff_plot(args.repair_glob)


if __name__ == "__main__":
    main()
