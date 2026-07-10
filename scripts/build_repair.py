# -*- coding: utf-8 -*-
# 读 CFU csv，给每行打修复 action（keep/mask/replace），输出 repair csv + 诊断指标。
#
# 选择(selection) 和 动作(action) 解耦：
#   --selection percentile  : low CFU = CFU_delete < percentile(low_pct)
#   --selection raw_negative : low CFU = CFU_delete < 0（自然阈值，不依赖人为比例）
#   --selection validation_threshold : 占位（需 valid 集选，见 collect_results）
#   --action mask    : 低 CFU 行 -> 置 0
#   --action replace : 低 CFU 行 -> 用 pred_item 替换
#   --tail_protect   : tail 且 CFU 在中间 ambiguous 区间 -> keep（不修）
#
# 诊断指标（打到 stdout/log）：
#   repair_count, detected_noise_count, repaired_clean_count,
#   noise_precision, noise_recall,
#   clean_false_repair_rate, tail_false_repair_rate, head_false_repair_rate

import os
import csv
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", required=True)
    ap.add_argument("--selection", default="percentile",
                    choices=["percentile", "raw_negative", "validation_threshold"])
    ap.add_argument("--low_pct", type=float, default=20.0)
    ap.add_argument("--action", default="mask", choices=["mask", "replace"])
    ap.add_argument("--tail_protect", type=int, default=0, help="1=tail ambiguous 保护")
    ap.add_argument("--ambig_lo", type=float, default=40.0)
    ap.add_argument("--ambig_hi", type=float, default=60.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="直接指定 CFU 阈值（优先于 selection）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.cfu_csv)
    c = df["CFU_delete"].values
    is_tail = df["is_tail"].values.astype(bool)
    is_noise = df["is_injected_noise"].values.astype(bool)
    has_pred = "pred_item" in df.columns
    pred = df["pred_item"].values if has_pred else df["last_item"].values

    # 选阈值
    if args.threshold is not None:
        low_thr = float(args.threshold)
        sel_name = f"thr{low_thr:.3f}"
    elif args.selection == "raw_negative":
        low_thr = 0.0
        sel_name = "raw_negative"
    elif args.selection == "validation_threshold":
        # 占位：先按 percentile 跑，真正 valid 选择在 collect_results 做
        low_thr = np.percentile(c, args.low_pct)
        sel_name = f"valpct{args.low_pct}"
    else:  # percentile
        low_thr = np.percentile(c, args.low_pct)
        sel_name = f"pct{args.low_pct}"

    amb_lo = np.percentile(c, args.ambig_lo)
    amb_hi = np.percentile(c, args.ambig_hi)

    actions, replace_items = [], []
    for i in range(len(df)):
        ci = c[i]
        tail = is_tail[i]
        ambig = (ci >= amb_lo) and (ci <= amb_hi)
        low = ci < low_thr
        if args.tail_protect and ambig and tail:
            act, rep = "keep", ""          # tail 不确定 -> 保护
        elif low:
            if args.action == "replace":
                act, rep = "replace", int(pred[i])
            else:
                act, rep = "mask", ""
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

    # ===== 诊断指标 =====
    repair_mask = np.array([a != "keep" for a in actions])
    total_noise = is_noise.sum()
    total_clean = (~is_noise).sum()
    real_tail = is_tail & (~is_noise)
    real_head = (~is_tail) & (~is_noise)
    detected_noise = (repair_mask & is_noise).sum()
    repaired_clean = (repair_mask & (~is_noise)).sum()
    repair_count = int(repair_mask.sum())

    noise_precision = detected_noise / repair_count if repair_count else 0.0
    noise_recall = detected_noise / total_noise if total_noise else 0.0
    clean_frr = repaired_clean / total_clean if total_clean else 0.0
    tail_frr = (repair_mask & real_tail).sum() / real_tail.sum() if real_tail.sum() else 0.0
    head_frr = (repair_mask & real_head).sum() / real_head.sum() if real_head.sum() else 0.0

    print(f"[build_repair] selection={sel_name} action={args.action} tail_protect={args.tail_protect}")
    print(f"  low_thr={low_thr:.4f}  repair_count={repair_count}/{len(df)}")
    print(f"  noise_precision={noise_precision:.4f}  noise_recall={noise_recall:.4f}  (detected {detected_noise}/{total_noise})")
    print(f"  clean_false_repair_rate={clean_frr:.4f}  (repaired_clean={repaired_clean})")
    print(f"  tail_false_repair_rate={tail_frr:.4f}  head_false_repair_rate={head_frr:.4f}")

    # 也写一份指标到 json 方便 collect
    import json
    metrics = {
        "selection": sel_name, "action": args.action, "tail_protect": args.tail_protect,
        "low_thr": float(low_thr), "repair_count": repair_count,
        "noise_precision": float(noise_precision), "noise_recall": float(noise_recall),
        "detected_noise": int(detected_noise), "repaired_clean": int(repaired_clean),
        "clean_false_repair_rate": float(clean_frr),
        "tail_false_repair_rate": float(tail_frr),
        "head_false_repair_rate": float(head_frr),
    }
    with open(args.out.replace(".csv", ".json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
