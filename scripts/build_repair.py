# -*- coding: utf-8 -*-
# 读 CFU csv，给每行打修复 action（keep/mask/replace），输出 repair csv。
#
# 策略 --strategy:
#   cfu_mask         : 低 CFU -> mask（置 0），其余 keep          （v1 命门测试）
#   cfu_mask_tail    : 低 CFU+head -> mask；低 CFU+tail -> replace；ambiguous tail -> keep
#   cfu_replace_tail : 低 CFU -> replace（用 pred_item）；ambiguous tail -> keep
#
# 低 CFU = CFU_delete < percentile(low_pct)。replace 需 CFU csv 含 pred_item 列。

import os
import csv
import argparse
from collections import Counter

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["cfu_mask", "cfu_mask_tail", "cfu_replace_tail"])
    ap.add_argument("--low_pct", type=float, default=20.0)
    ap.add_argument("--ambig_lo", type=float, default=40.0)
    ap.add_argument("--ambig_hi", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cfu_csv)
    c = df["CFU_delete"].values
    low_thr = np.percentile(c, args.low_pct)
    amb_lo = np.percentile(c, args.ambig_lo)
    amb_hi = np.percentile(c, args.ambig_hi)
    is_tail = df["is_tail"].values.astype(bool)
    is_noise = df["is_injected_noise"].values.astype(bool)
    has_pred = "pred_item" in df.columns
    pred = df["pred_item"].values if has_pred else df["last_item"].values

    actions, replace_items = [], []
    for i in range(len(df)):
        ci = c[i]
        tail = is_tail[i]
        ambig = (ci >= amb_lo) and (ci <= amb_hi)

        if args.strategy == "cfu_mask":
            act = "mask" if ci < low_thr else "keep"
            rep = ""
        elif args.strategy == "cfu_mask_tail":
            if ci < low_thr:
                if tail:
                    act, rep = "replace", int(pred[i])
                else:
                    act, rep = "mask", ""
            elif ambig and tail:
                act, rep = "keep", ""   # tail 保护
            else:
                act, rep = "keep", ""
        else:  # cfu_replace_tail
            if ci < low_thr:
                act, rep = "replace", int(pred[i])
            elif ambig and tail:
                act, rep = "keep", ""
            else:
                act, rep = "keep", ""
        actions.append(act)
        replace_items.append(rep)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "action", "replace_item"])
        for i, r in enumerate(df["row"].values):
            w.writerow([int(r), actions[i], replace_items[i]])

    repair_mask = np.array([a != "keep" for a in actions])
    print(f"[build_repair] {args.strategy}: {Counter(actions)}")
    print(f"  low_thr={low_thr:.4f}  repaired={repair_mask.sum()}/{len(df)}")
    print(f"  repaired 中是噪声的: {(repair_mask & is_noise).sum()}/{is_noise.sum()} (recall on noise)")
    # 误伤：被 repair 但不是噪声的
    fp = (repair_mask & (~is_noise)).sum()
    print(f"  repaired 中是真实交互的(误伤): {fp}")


if __name__ == "__main__":
    main()
