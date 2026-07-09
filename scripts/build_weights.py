# -*- coding: utf-8 -*-
# 读 CFU csv，生成 per-row 训练权重文件 (row, weight)。
#
# 策略 --strategy:
#   cfu_only   : weight = sigmoid((CFU - mean) / tau)            高 CFU -> 高权重
#   loss_reweight: weight = sigmoid(-(loss - mean) / tau)        高 loss -> 低权重（普通去噪 baseline）
#   cfu_tail   : cfu_only + tail ambiguous 保护
#                tail item 且 CFU 在中间 20% ambiguous 区间 -> weight = max(weight, tail_min)
#
# 用法: python scripts/build_weights.py --cfu_csv logs/cfu_xxx.csv \
#            --strategy cfu_tail --tau 0.2 --tail_min 0.5 --out logs/w_cfu_tail.csv

import os
import csv
import argparse

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["cfu_only", "loss_reweight", "cfu_tail"])
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--tail_min", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cfu_csv)

    if args.strategy == "cfu_only":
        c = df["CFU_delete"].values
        w = sigmoid((c - c.mean()) / args.tau)
    elif args.strategy == "loss_reweight":
        c = df["orig_loss"].values
        w = sigmoid(-(c - c.mean()) / args.tau)
    elif args.strategy == "cfu_tail":
        c = df["CFU_delete"].values
        w = sigmoid((c - c.mean()) / args.tau)
        # ambiguous = CFU 在中间 20%（即 |CFU - mean| 较小）
        lo = np.percentile(c, 40)
        hi = np.percentile(c, 60)
        is_tail = df["is_tail"].values.astype(bool)
        ambig = (c >= lo) & (c <= hi)
        protect = is_tail & ambig
        w = np.where(protect, np.maximum(w, args.tail_min), w)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "weight"])
        for r, weight in zip(df["row"].values, w):
            writer.writerow([int(r), float(weight)])
    print(f"[build_weights] {args.strategy}: wrote {len(w)} weights -> {args.out}")
    print(f"  weight stats: mean={w.mean():.4f} min={w.min():.4f} max={w.max():.4f}")

    # TMR 预统计：真实 tail 交互被赋低权重(<0.3)的比例
    is_tail = df["is_tail"].values.astype(bool)
    is_noise = df["is_injected_noise"].values.astype(bool)
    real_tail = is_tail & (~is_noise)
    if real_tail.sum() > 0:
        tmr = (w[real_tail] < 0.3).mean()
        print(f"  TMR(real_tail, weight<0.3) = {tmr:.4f}  (n={real_tail.sum()})")
    if is_noise.sum() > 0:
        noise_down = (w[is_noise] < 0.3).mean()
        print(f"  noise down-weighted(<0.3) ratio = {noise_down:.4f}  (n={is_noise.sum()})")


if __name__ == "__main__":
    main()
